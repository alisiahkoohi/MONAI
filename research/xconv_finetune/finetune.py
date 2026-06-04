"""Orchestration: size the GPU, finetune, verify conv updates, record memory.

``run(cfg)`` is shared by both drivers. The protocol that realizes the user's
request:

1. Pick the batch that fits the budget *for the baseline* (so both methods use the
   same batch — a fair memory comparison).
2. For XConv, pick the largest probing count ``ps`` that still fits at that batch
   (spend the freed activation memory on a less-noisy gradient).
3. Finetune all layers; snapshot conv weights before/after to prove the convs
   train. Record the peak memory of the whole loop.

Nothing here changes the optimization other than how the conv backward is
computed, so baseline vs. XConv is a controlled comparison.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import data as datamod
import memory as mem
from config import Config
from engine import (build_loss, make_optimizer, make_synthetic_step,
                    finetune, evaluate_dice)
from model import (build_segresnet, apply_xconv, conv_trainable_report,
                   snapshot_conv_weights, conv_update_deltas,
                   load_pretrained, assert_convert_preserved)


def _batch_sidecar(cfg: Config) -> str:
    """Path that records the baseline-chosen batch, so XConv reuses it exactly."""
    return os.path.join(cfg.out_dir, f"_chosen_batch_p{cfg.patch}_f{cfg.init_filters}.json")


def _resolve_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; falling back to CPU (memory study is GPU-only).")
        return torch.device("cpu")
    return torch.device(cfg.device)


def _baseline_builder(cfg: Config, device: torch.device):
    return lambda: build_segresnet(datamod.IN_CHANNELS, datamod.OUT_CHANNELS,
                                   cfg.init_filters, cfg.seed, device)


def _xconv_builder(cfg: Config, device: torch.device):
    def build(ps: int):
        model = build_segresnet(datamod.IN_CHANNELS, datamod.OUT_CHANNELS,
                                cfg.init_filters, cfg.seed, device)
        return apply_xconv(model, ps=ps, xmode=cfg.xmode, target=cfg.xconv_target)
    return build


def run(cfg: Config) -> dict:
    device = _resolve_device(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)
    loss_fn = build_loss()
    step_fn = make_synthetic_step(loss_fn, cfg.lr, cfg.weight_decay, cfg.patch, device)

    # 1) The baseline max batch B0 is the reference, and the batch ceiling for
    #    XConv. It is persisted so the (independent, stochastic) XConv process
    #    sees the same B0.
    sidecar = _batch_sidecar(cfg)
    b0 = 0
    if os.path.exists(sidecar):
        b0 = json.load(open(sidecar)).get("batch_size", 0)
    if b0 <= 0:
        ceil = cfg.batch_size if cfg.batch_size > 0 else 16
        cands = tuple(b for b in (1, 2, 3, 4, 6, 8, 12, 16) if b <= ceil)
        print(f"[sizing] finding baseline max batch B0 (patch={cfg.patch}, "
              f"budget={cfg.mem_budget_gb} GB)")
        b0, base_mem = mem.find_max_batch(_baseline_builder(cfg, device), step_fn,
                                          device, cfg.mem_budget_gb, candidates=cands)
        json.dump({"batch_size": b0, "baseline_peak_gb": base_mem}, open(sidecar, "w"))
        print(f"[sizing] baseline B0={b0} (peak {base_mem:.2f} GB)")

    # 2) Resolve the (batch, ps) actually used.
    if cfg.method == "baseline":
        if cfg.batch_size <= 0:
            cfg.batch_size = b0
    else:  # xconv
        xb = _xconv_builder(cfg, device)
        if cfg.size_strategy == "max_ps" and cfg.ps <= 0:
            # Maximize r; drop batch toward 1 only as far as a larger r requires.
            print(f"[sizing] maximizing r (drop batch as needed, ceiling B0={b0})")
            cfg.ps, cfg.batch_size, ps_mem = mem.maximize_ps(
                xb, step_fn, device, cfg.mem_budget_gb, max_batch=b0)
            drop = "" if cfg.batch_size == b0 else f" (batch dropped {b0}->{cfg.batch_size} for larger r)"
            print(f"[sizing] chosen r=ps={cfg.ps} at batch={cfg.batch_size} "
                  f"(peak {ps_mem:.2f} GB){drop}")
        else:  # fixed_batch, or an explicit ps/batch override
            if cfg.batch_size <= 0:
                cfg.batch_size = b0
            if cfg.ps <= 0:
                print(f"[sizing] finding max r=ps at fixed batch={cfg.batch_size}")
                cfg.ps, ps_mem = mem.find_max_ps(
                    xb, step_fn, cfg.batch_size, device, cfg.mem_budget_gb)
                print(f"[sizing] chosen r=ps={cfg.ps} (peak {ps_mem:.2f} GB)")

    # 3) build the model actually used for finetuning.
    #    CRITICAL ORDER: build -> load pretrained -> convert, so that XConv
    #    conversion preserves the pretrained initialization (it reuses each
    #    conv's weight Parameter; the guard below verifies nothing changed).
    model = _baseline_builder(cfg, device)()                 # plain SegResNet
    preserved_delta = None
    if cfg.pretrained_ckpt:
        info = load_pretrained(model, cfg.pretrained_ckpt)
        print(f"[init] loaded pretrained from {cfg.pretrained_ckpt}: {info['loaded']} "
              f"tensors (skipped {len(info['skipped_shape_mismatch'])} shape-mismatched "
              f"e.g. output head)")
    else:
        print("[init] no pretrained_ckpt -> random init "
              "(run pretrain.py first for a genuine finetuning init)")
    if cfg.method == "xconv":
        pre_convert = snapshot_conv_weights(model)           # the init to protect
        apply_xconv(model, ps=cfg.ps, xmode=cfg.xmode, target=cfg.xconv_target)
        preserved_delta = assert_convert_preserved(model, pre_convert)
        print(f"[guard] convert_net preserved the loaded init "
              f"(max rel. conv-weight delta {preserved_delta:.1e})")
    report = conv_trainable_report(model)
    print(f"[model] {cfg.method}: {report}")

    loader = datamod.train_loader(cfg.data_dir, cfg.patch, cfg.batch_size,
                                  cfg.cache_num, cfg.num_workers)
    optimizer = make_optimizer(model, cfg.lr, cfg.weight_decay)

    # cuDNN autotuning only matters once the shape is fixed (post-sizing).
    torch.backends.cudnn.benchmark = True

    # snapshot conv weights -> finetune -> measure the change (proves convs train).
    before = snapshot_conv_weights(model)
    print(f"[finetune] {cfg.method} for {cfg.max_steps} steps "
          f"(batch={cfg.batch_size}" + (f", ps={cfg.ps}" if cfg.method == "xconv" else "") + ")")
    hist = finetune(model, loader, loss_fn, optimizer, cfg.max_steps, device,
                    mem.track_peak(device))
    deltas = conv_update_deltas(model, before)
    dvals = list(deltas.values())
    # A conv is "trained by the gradient" only if its update dwarfs AdamW's
    # weight-decay-only drift (~lr*wd*steps), which would move weights even with a
    # zero gradient. The boolean all(>0) is therefore NOT a valid proof; we
    # require the update to exceed that drift by a wide margin.
    wd_drift = cfg.lr * cfg.weight_decay * cfg.max_steps
    gate = max(1e-3, 50.0 * wd_drift)
    conv_update = {
        "min_rel_update": min(dvals), "max_rel_update": max(dvals),
        "mean_rel_update": sum(dvals) / len(dvals),
        "wd_only_drift": wd_drift, "gradient_gate": gate,
        "convs_gradient_trained": bool(min(dvals) > gate),
    }
    print(f"[verify] convs gradient-trained: {conv_update['convs_gradient_trained']} "
          f"(min rel. update {conv_update['min_rel_update']:.3e} vs gate {gate:.1e}; "
          f"wd-only drift {wd_drift:.1e})")

    dice = None
    del optimizer  # drop AdamW moment buffers before the (memory-heavy) eval
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        val_ds = datamod.val_dataset(cfg.data_dir, cfg.num_workers)
        dice = evaluate_dice(model, val_ds, cfg.val_volumes, cfg.patch, device)
        print(f"[eval] mean foreground Dice on {cfg.val_volumes} val vols: {dice:.4f}")
    except RuntimeError as err:
        kind = "OOM during sliding-window eval" if "out of memory" in str(err).lower() else "error"
        print(f"[eval] skipped ({kind}): {err}")

    losses = hist["losses"]
    if losses:  # arrays go to .npy; JSON holds only the summary (house style)
        np.save(os.path.join(cfg.out_dir, cfg.run_name() + "_losses.npy"),
                np.asarray(losses, dtype=np.float32))
    result = {
        "config": cfg.as_dict(),
        "run_name": cfg.run_name(),
        "baseline_b0": b0,
        "resolved_batch": cfg.batch_size,
        "resolved_ps": cfg.ps if cfg.method == "xconv" else None,
        "pretrained_ckpt": cfg.pretrained_ckpt or None,
        "convert_preserved_max_delta": preserved_delta,
        "conv_report": report,
        "conv_update": conv_update,
        "peak_gb": hist["peak_gb"],
        "n_steps": len(losses),
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "val_dice": dice,
    }
    out_path = os.path.join(cfg.out_dir, cfg.run_name() + ".json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {cfg.method}: peak {hist['peak_gb']:.2f} GB -> saved {out_path}")
    return result
