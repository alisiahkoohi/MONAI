"""AGE + peak-memory of the spleen UNet: exact vs XConv, swept over probing count r.

Produces the two canonical rad-vs-xconv figures for THIS example (the pretrained
MONAI spleen UNet), reusing the in-house metrics verbatim so the numbers match the
paper methodology:

* **AGE** (paper Eq. 9) via ``radcompare.age`` — exact full-dataset gradient vs
  minibatch gradient on the conv weights. The exact model's AGE is the sampling
  floor; XConv's AGE should decay toward it as r grows (the `ali` boundary fix
  makes XConv unbiased — no residual plateau).
* **Peak memory** via ``radcompare.memory.peak_memory_mib`` — ``torch_peak`` from
  the repo ``MemoryTracker`` with the 2-iteration warm-up protocol.

Both at a fixed patch + batch (the bundle's native 96^3); AGE uses a FIXED set of
spleen patches (cropped once) so the "full gradient" is well defined; peak memory
uses random 3D inputs of the same shape. Figures use the repo paper theme.

GPU required to run (peak memory uses CUDA); everything is sized small. CPU-only
import/AGE smoke is possible with --device cpu --skip_memory.

    python age_memory.py --rs 2,4,8,16,32,64,128,256 --batch 8 --subset 64 --n_runs 3
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from radcompare import average_gradient_error, exact_full_gradient, peak_memory_mib
from radcompare import plotting

import bundle as B
import xconv_ops as xc

_HERE = os.path.dirname(os.path.abspath(__file__))
EXACT_COLOR = "#1f77b4"
XCONV_COLORS = ["#fdae6b", "#fd8d3c", "#e6550d", "#d62728", "#a50f15"]


def conv_weight_names(model) -> list[str]:
    """Full parameter names of every Conv3d weight (shared by exact and XConv)."""
    return [n + ".weight" for n, _ in xc.conv_layers(model)]


def fixed_patch_loader(dataset_dir, patch, gpu_batch, n_patches, num_workers, synthetic=False):
    """A fixed in-memory loader of ``n_patches`` (image,label) patches.

    AGE needs a fixed dataset; the bundle's RandCrop is stochastic, so we
    materialize the patches once and serve them deterministically. ``synthetic``
    uses random tensors at ``patch`` (for fast CPU smoke); otherwise real spleen
    patches are sourced from the bundle (always its native 96^3 crop).
    """
    from torch.utils.data import DataLoader, TensorDataset
    if synthetic:
        img = torch.randn(n_patches, B.IN_CHANNELS, patch, patch, patch)
        lbl = torch.randint(0, B.OUT_CHANNELS, (n_patches, 1, patch, patch, patch)).float()
    else:
        src = B.train_loader(dataset_dir, B.NUM_SAMPLES, num_workers)  # valid source batch
        imgs, lbls, got = [], [], 0
        for batch in src:
            imgs.append(batch["image"]); lbls.append(batch["label"])
            got += batch["image"].shape[0]
            if got >= n_patches:
                break
        img = torch.cat(imgs)[:n_patches].contiguous()
        lbl = torch.cat(lbls)[:n_patches].contiguous()
    return DataLoader(TensorDataset(img, lbl), batch_size=gpu_batch, shuffle=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rs", default="2,4,8,16,32,64,128,256")
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--batch", type=int, default=8)        # GPU batch for AGE + peak
    p.add_argument("--subset", type=int, default=64)      # fixed patches for the AGE dataset
    p.add_argument("--n_runs", type=int, default=3)       # AGE repeats (probe re-draws) for +/-sigma
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dataset_dir", default=os.path.join(_HERE, "data", "Task09_Spleen"))
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results", "age_memory"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_memory", action="store_true")  # CPU AGE smoke without CUDA tracker
    p.add_argument("--synthetic", action="store_true")    # random patches (fast CPU smoke)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rs = [int(x) for x in a.rs.split(",")]
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if not a.synthetic and a.patch != B.PATCH:
        print(f"[note] real spleen data uses the bundle's native {B.PATCH}^3 crop; "
              f"overriding --patch {a.patch} -> {B.PATCH}")
        a.patch = B.PATCH
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    loss_fn = B.loss_fn(a.dataset_dir)

    # shared theta = the pretrained spleen weights (the finetuning init)
    exact = B.pretrained_net(a.dataset_dir, device)
    names = conv_weight_names(exact)
    loader = fixed_patch_loader(a.dataset_dir, a.patch, a.batch, a.subset, a.num_workers,
                                synthetic=a.synthetic)

    # random 3D inputs for the peak-memory protocol (values irrelevant to memory)
    rand_x = torch.randn(a.batch, B.IN_CHANNELS, a.patch, a.patch, a.patch, device=device)
    rand_y = torch.randint(0, B.OUT_CHANNELS, (a.batch, 1, a.patch, a.patch, a.patch),
                           device=device, dtype=torch.float32)

    print(f"[age] exact full-dataset gradient over {a.subset} patches ...")
    full_grad = exact_full_gradient(exact, loader, names, str(device), loss_fn=loss_fn)
    age_exact = float(np.mean([average_gradient_error(exact, loader, full_grad, names,
                                                      str(device), loss_fn=loss_fn)
                               for _ in range(a.n_runs)]))
    peak_exact = (None if a.skip_memory else
                  peak_memory_mib(exact, rand_x, rand_y, loss_fn=loss_fn))
    print(f"[exact] AGE(floor)={age_exact:.4e}"
          + ("" if peak_exact is None else f" | peak={peak_exact:.0f} MiB"))
    del exact
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for r in rs:
        xnet = xc.apply_xconv(B.pretrained_net(a.dataset_dir, device), r, "independent", "conv")
        ages = [average_gradient_error(xnet, loader, full_grad, names, str(device), loss_fn=loss_fn)
                for _ in range(a.n_runs)]
        peak = (None if a.skip_memory else
                peak_memory_mib(xnet, rand_x, rand_y, loss_fn=loss_fn))
        rows.append({"r": r, "age_mean": float(np.mean(ages)), "age_std": float(np.std(ages)),
                     "peak_mib": peak})
        print(f"[xconv r={r:<4}] AGE={np.mean(ages):.4e} +/- {np.std(ages):.1e}"
              + ("" if peak is None else f" | peak={peak:.0f} MiB"))
        del xnet
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {"patch": a.patch, "batch": a.batch, "subset": a.subset, "n_runs": a.n_runs,
              "age_exact": age_exact, "peak_exact_mib": peak_exact, "rows": rows}
    json.dump(result, open(os.path.join(a.out_dir, "age_memory.json"), "w"), indent=2)
    _plots(result, a.out_dir)
    print(f"[done] -> {a.out_dir}")


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path):
    fig.savefig(path + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {path}.pdf/.png")


def _plots(result, out_dir):
    plotting.apply_paper_style()
    rs = [row["r"] for row in result["rows"]]

    # AGE vs r: XConv decays toward the exact sampling floor (unbiased)
    fig, ax = plt.subplots(figsize=(5, 4))
    age = [row["age_mean"] for row in result["rows"]]
    std = [row["age_std"] for row in result["rows"]]
    ax.fill_between(rs, np.subtract(age, std), np.add(age, std), color=EXACT_COLOR, alpha=0.15)
    ax.plot(rs, age, "o-", color=XCONV_COLORS[2], label="XConv")
    ax.axhline(result["age_exact"], ls="--", color=EXACT_COLOR, label="exact (sampling floor)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("probing vectors $r$"); ax.set_ylabel("Average Gradient Error")
    ax.legend(frameon=False); _despine(ax)
    _save(fig, os.path.join(out_dir, "age_vs_r"))

    # Peak memory vs r: XConv vs the exact baseline (the <= baseline line)
    if result["peak_exact_mib"] is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        peak = [row["peak_mib"] for row in result["rows"]]
        ax.plot(rs, peak, "o-", color=XCONV_COLORS[2], label="XConv")
        ax.axhline(result["peak_exact_mib"], ls="--", color=EXACT_COLOR, label="exact baseline")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("probing vectors $r$"); ax.set_ylabel("peak memory (MiB)")
        ax.legend(frameon=False); _despine(ax)
        _save(fig, os.path.join(out_dir, "peak_memory_vs_r"))


if __name__ == "__main__":
    main()
