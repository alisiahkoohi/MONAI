"""Pretrain a plain SegResNet on Spleen and save a checkpoint to finetune FROM.

This produces the ``pretrained_ckpt`` that ``train_baseline.py`` / ``train_xconv.py``
load, so the memory study is a genuine *finetuning* (convs start from a trained
init, not random) and the XConv conversion's init-preservation is meaningful.

The checkpoint is a plain SegResNet ``state_dict`` (regular Conv3d), so it loads
identically into the baseline and into the to-be-converted XConv model.

Usage:
    python pretrain.py --pretrain_steps 200 --patch 96
    # then:
    python train_baseline.py --pretrained_ckpt results/pretrained_segresnet_p96_f16.pth
    python train_xconv.py    --pretrained_ckpt results/pretrained_segresnet_p96_f16.pth
"""
from __future__ import annotations

import os

import torch

import data as datamod
import memory as mem
from config import parse_args
from engine import build_loss, make_optimizer, make_synthetic_step, finetune
from model import build_segresnet
from finetune import _resolve_device


def default_ckpt_path(cfg) -> str:
    return os.path.join(cfg.out_dir,
                        f"pretrained_segresnet_p{cfg.patch}_f{cfg.init_filters}.pth")


def main() -> None:
    cfg = parse_args(method="baseline")
    device = _resolve_device(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)
    loss_fn = build_loss()

    if cfg.batch_size <= 0:
        step_fn = make_synthetic_step(loss_fn, cfg.lr, cfg.weight_decay, cfg.patch, device)
        cfg.batch_size, _ = mem.find_max_batch(
            lambda: build_segresnet(datamod.IN_CHANNELS, datamod.OUT_CHANNELS,
                                    cfg.init_filters, cfg.seed, device),
            step_fn, device, cfg.mem_budget_gb)
    print(f"[pretrain] batch_size={cfg.batch_size}, steps={cfg.pretrain_steps}")

    torch.backends.cudnn.benchmark = True
    model = build_segresnet(datamod.IN_CHANNELS, datamod.OUT_CHANNELS,
                            cfg.init_filters, cfg.seed, device)
    loader = datamod.train_loader(cfg.data_dir, cfg.patch, cfg.batch_size,
                                  cfg.cache_num, cfg.num_workers)
    optimizer = make_optimizer(model, cfg.lr, cfg.weight_decay)
    hist = finetune(model, loader, loss_fn, optimizer, cfg.pretrain_steps, device,
                    mem.track_peak(device))

    out = cfg.pretrained_ckpt or default_ckpt_path(cfg)
    torch.save({"state_dict": model.state_dict(),
                "config": cfg.as_dict(),
                "final_loss": hist["losses"][-1] if hist["losses"] else None}, out)
    print(f"[pretrain] saved {out} (final loss "
          f"{hist['losses'][-1] if hist['losses'] else float('nan'):.4f})")


if __name__ == "__main__":
    main()
