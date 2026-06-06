"""Publication figures for the XConv spleen-UNet finetuning study.

Uses MONAI's own visualization utilities (``blend_images``, ``matshow3d``):

* ``segmentation_panel`` — input CT | ground-truth overlay | prediction overlay on
  the most-spleen axial slice (the model is the bundle's pretrained UNet = the
  finetuning init; 60 finetune steps are negligible on this near-converged model).
* ``segmentation_3d`` — a grid of axial slices with the predicted overlay (matshow3d).
* ``comparison`` — peak memory and Dice, exact vs XConv, read from results/*.json.

    python visualize.py                 # all figures (seg needs GPU; comparison is CPU)
    python visualize.py --compare_only  # just the bar chart (CPU, from JSONs)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.visualize import blend_images, matshow3d

import bundle as B
import xconv_ops as xc  # noqa: F401 (kept for optional converted-model viz)

_HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": False, "font.size": 11})
EXACT, XCONV = "#1f77b4", "#d62728"


def _save(fig, path):
    fig.savefig(path + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}.pdf / .png", flush=True)


def _to_rgb(blend):  # (3,H,W) tensor -> (H,W,3) numpy in [0,1]
    a = blend.detach().cpu().numpy()
    a = np.moveaxis(a, 0, -1)
    return np.clip(a, 0, 1)


def segmentation_figures(dataset_dir, out_dir, device):
    model = B.pretrained_net(dataset_dir, device)
    model.eval()
    batch = next(iter(B.val_loader(dataset_dir, 1, 0)))
    img = batch["image"].to(device)           # (1,1,H,W,D)
    lbl = batch["label"][0].cpu().float()      # (1,H,W,D)
    with torch.no_grad():
        logits = sliding_window_inference(img, (96, 96, 96), 4, model, overlap=0.25)
    pred = torch.argmax(logits, dim=1, keepdim=True)[0].cpu().float()   # (1,H,W,D)
    ct = img[0].cpu().float()                                           # (1,H,W,D)

    # axial slice with the most spleen in the ground truth
    z = int(lbl[0].sum(dim=(0, 1)).argmax())
    ct_s, gt_s, pr_s = ct[:, :, :, z], lbl[:, :, :, z], pred[:, :, :, z]   # each (1,H,W)
    gt_b = blend_images(ct_s, gt_s, alpha=0.5, cmap="hsv", rescale_arrays=True)
    pr_b = blend_images(ct_s, pr_s, alpha=0.5, cmap="hsv", rescale_arrays=True)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.7))
    ax[0].imshow(ct_s[0].numpy().T, cmap="gray", origin="lower"); ax[0].set_title("input CT")
    ax[1].imshow(np.transpose(_to_rgb(gt_b), (1, 0, 2)), origin="lower"); ax[1].set_title("ground truth")
    ax[2].imshow(np.transpose(_to_rgb(pr_b), (1, 0, 2)), origin="lower"); ax[2].set_title("prediction")
    for a in ax:
        a.axis("off")
    _save(fig, os.path.join(out_dir, "segmentation_panel"))

    # 3D grid of slices with predicted overlay (only slices that contain spleen)
    zc = torch.where(pred[0].sum(dim=(0, 1)) > 0)[0]
    if len(zc) >= 4:
        lo, hi = int(zc.min()), int(zc.max())
        vol = blend_images(ct[:, :, :, lo:hi + 1], pred[:, :, :, lo:hi + 1],
                           alpha=0.5, cmap="hsv", rescale_arrays=True)  # (3,H,W,d)
        try:
            fig2 = plt.figure(figsize=(11, 8))
            matshow3d(volume=vol, fig=fig2, title="XConv-finetuned spleen segmentation (axial slices)",
                      frames_per_row=6, frame_dim=-1, channel_dim=0, every_n=1, cmap=None)
            _save(fig2, os.path.join(out_dir, "segmentation_3d"))
        except Exception as e:
            print(f"matshow3d failed ({e}); skipping 3D grid", flush=True)


def comparison_figure(res_dir, out_dir):
    bfs = glob.glob(os.path.join(res_dir, "baseline_unet_spleen_b*.json"))
    xfs = glob.glob(os.path.join(res_dir, "xconv_unet_spleen_b*.json"))
    if not bfs or not xfs:
        print("comparison: missing result JSONs", flush=True)
        return
    b = json.load(open(sorted(bfs, key=os.path.getmtime)[-1]))
    bb = b.get("resolved_batch")
    xs = [d for d in (json.load(open(f)) for f in xfs)
          if "peak_mib" in d and d.get("resolved_batch") == bb]
    xs.sort(key=lambda d: d.get("resolved_ps") or 0)
    labels = ["exact"] + [f"XConv\nr={x['resolved_ps']}, {x['n_steps']} st." for x in xs]
    mems = [b["peak_mib"]] + [x["peak_mib"] for x in xs]
    dices = [b.get("val_dice") or 0] + [x.get("val_dice") or 0 for x in xs]
    # exact blue; XConv reds, darker = smaller r (more saving)
    reds = ["#fb6a4a", "#de2d26", "#a50f15", "#67000d"]
    colors = [EXACT] + [reds[min(i, len(reds) - 1)] for i in range(len(xs))]

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax[0].bar(labels, mems, color=colors, width=0.7)
    ax[0].axhline(b["peak_mib"], ls="--", lw=0.8, color=EXACT)
    ax[0].set_ylabel("peak memory (MiB)"); ax[0].set_title("memory")
    ax[1].bar(labels, dices, color=colors, width=0.7)
    ax[1].axhline(b.get("val_dice") or 0, ls="--", lw=0.8, color=EXACT)
    ax[1].set_ylabel("foreground Dice"); ax[1].set_ylim(0, 1); ax[1].set_title("accuracy")
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"spleen UNet finetune at batch {b.get('resolved_batch')}", y=1.02, fontsize=11)
    _save(fig, os.path.join(out_dir, "comparison"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default=os.path.join(_HERE, "data", "Task09_Spleen"))
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results", "figures"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--compare_only", action="store_true")
    p.add_argument("--seg_only", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if not a.compare_only:
        dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
        segmentation_figures(a.dataset_dir, a.out_dir, dev)
    if not a.seg_only:
        comparison_figure(os.path.join(_HERE, "results"), a.out_dir)


if __name__ == "__main__":
    main()
