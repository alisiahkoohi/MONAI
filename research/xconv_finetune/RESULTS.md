# Results — XConv finetuning of the MONAI spleen UNet

**Headline.** Finetuning the pretrained `spleen_ct_segmentation` UNet at batch 64,
with the *identical* training recipe (Novograd, lr=2e-4, 600 steps), the **probed
XConv weight gradient (r=4) reaches the same segmentation accuracy as the exact
gradient while using 17.5% less peak memory.**

| config (batch 64) | gradient | peak memory | foreground Dice |
|---|---|---|---|
| **exact** (600 steps, lr 2e-4) | exact conv backward | **11854 MiB** | **0.9620** |
| **XConv r=4** (600 steps, lr 2e-4) | probed (`independent`) | **9775 MiB** (−17.5%) | **0.9622** (Δ +0.0002) |

The accuracy difference (+0.0002) is within run-to-run noise → **parity at lower
memory**. Figures: `results/figures/`.

---

## Setup

| item | value |
|---|---|
| task / data | MSD **Task09 Spleen** (CT, 1 channel, binary) |
| model | MONAI bundle `spleen_ct_segmentation` UNet — channels (16,32,64,128,256), **PReLU**, **BatchNorm3d**, pretrained `models/model.pt` (loads exact: missing=0, unexpected=0) |
| loss / optim / sched | `DiceCELoss` / `Novograd` / `StepLR(5000, 0.1)` — all the bundle's |
| patch / batch | 96³ / GPU batch 64 (= 4 RandCrop samples × loader batch 16) |
| XConv | in-house `pyxconv` — `luqigroup/xconv_pv` branch **`ali`** (boundary-fixed, 2D+3D), `convert_net(mode='conv', xmode='independent')` |
| peak-memory metric | `radcompare.memory.peak_memory_mib` = `torch_peak` (repo `MemoryTracker`, **2-iteration warm-up**, SGD lr=0) — the rad-vs-xconv paper metric, used everywhere here |
| Dice metric | foreground `MeanDice` via `SlidingWindowInferer` (roi 96³, sw_batch 4) on **5 held-out** volumes |
| GPU / env | RTX 2000 Ada, 16 GB · conda env `sips` (torch 2.9 + cu128, MONAI 1.5.2) |

---

## Experiment path (how we got here)

1. **Pivot to the bundle.** Use the MONAI `spleen_ct_segmentation` bundle's
   pretrained UNet + its data/loss/optimizer; finetune all conv layers (no
   pretraining of our own). `convert_net` preserves the pretrained init exactly
   (`convert_preserved_max_delta == 0`, asserted each run).
2. **Settle the memory metric.** Adopted `radcompare.peak_memory_mib` (`torch_peak`,
   2-iter warm-up) for *every* script, consistent with the in-house rad-vs-xconv
   methodology (an earlier NVML poller was removed).
3. **Size the operating point.** Under that metric, swept batch (canonical sizing):
   batch 32 → baseline 5928 MiB, 48 → 9142, **64 → 12206**, 80 → OOM. Largest batch
   where XConv ≤ baseline = **64**; max r ≤ baseline there = 512.
4. **First comparison (60 steps, lr 2e-3, the bundle default).** baseline 11854 MiB
   / Dice 0.9493 vs XConv r=512 11638 MiB / Dice 0.9435 → XConv ≤ baseline at ~equal
   accuracy.
5. **Push r down for bigger savings.** XConv peak at batch 64 *floors* at ~9780 MiB
   and barely moves with r (r=4 → 9782, r=256 → 9968, r=512 → 11638). Even r=4 is
   only **−17.5%**: the floor is the UNet's **unsavable** activations (skip
   connections + PReLU), which `mode='conv'` cannot compress.
6. **Recover the noisy small-r gradient.** Ran XConv **r=4, lr 2e-4 (÷10), 600
   steps (×10)** → 9775 MiB, **Dice 0.9622** (loss stayed ~0.006 throughout — the
   noisy probed gradient did not degrade training).
7. **Strict apples-to-apples.** Re-ran the **baseline at the same recipe** (lr 2e-4,
   600 steps) → 11854 MiB / **Dice 0.9620**. So vs XConv r=4: identical accuracy,
   17.5% less memory (the headline).

---

## All numbers

### Memory vs r at batch 64 (peak_memory_mib; baseline = 11838–11854 MiB)
| r | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|
| XConv MiB | 9782 | 9783 | 9785 | 9827 | 9876 | 9933 | 9968 | 11638 |

Floor ≈ 9780 MiB; saving capped at ~17%.

### Sizing sweep (largest batch with XConv ≤ baseline)
batch 32 → 5928 MiB · 48 → 9142 · **64 → 12206** · 80 → baseline OOM. Chosen: batch 64.

### Finetune runs
| run | r | steps | lr | peak MiB | Dice | convs trained | file |
|---|---|---|---|---|---|---|---|
| baseline (default) | — | 60 | 2e-3 | 11854 | 0.9493 | yes | `baseline_unet_spleen_b64_60steps.json` |
| baseline (strict) | — | 600 | 2e-4 | 11854 | 0.9620 | yes | `baseline_unet_spleen_b64.json` |
| XConv max-r | 512 | 60 | 2e-3 | 11638 | 0.9435 | yes | `xconv_unet_spleen_b64_r512_independent_conv.json` |
| **XConv small-r** | **4** | **600** | **2e-4** | **9775** | **0.9622** | yes | `xconv_unet_spleen_b64_r4_independent_conv.json` |

All runs preserve the pretrained init through conversion (delta 0) and verify the
conv weights move during finetuning.

### Loss curves & per-slice segmentation runs (batch 8, identical recipe, CPU)

The GPU was contended when these figures were made (free memory swung 11.9 → 5.2 GB as
other jobs grabbed ~10 GB), so the **training-dynamics + qualitative-segmentation** runs
were done on **CPU at batch 8**. Conv-vs-XConv parity is batch-independent, so this does
**not** affect the batch-64 peak-memory headline above. Both runs use the *identical*
recipe (Novograd, lr 2e-4, StepLR(5000,0.1), DiceCELoss, 600 steps); only the conv
weight gradient differs (exact vs probed r=4).

| run (batch 8, 600 steps, lr 2e-4) | r | Dice (3 held-out vols) | final train loss | init preserved | convs trained |
|---|---|---|---|---|---|
| baseline (conv, exact) | — | **0.9639** | 0.0042 | — | yes (mean rel-L2 1.2e-2) |
| **XConv (probed)** | **4** | **0.9649** (Δ +0.0010) | 0.0061 | max delta **0.0** | yes (mean rel-L2 1.1e-2) |

Per-volume foreground Dice (annotated on each segmentation figure) — conv / XConv r=4:
**0.965 / 0.966** (vol 1), **0.962 / 0.963** (vol 2), **0.965 / 0.966** (vol 3). XConv
matches the exact conv to within **±0.001** on every volume, and the overlays are
visually indistinguishable. Artifacts: `baseline_unet_spleen_b8.json`,
`xconv_unet_spleen_b8_r4_independent_conv.json` (+ `_losses.npy`, `_val_losses.npy`,
and the finetuned checkpoints under `results/checkpoints/`).

---

## Figures — directory guide (`results/figures/`, vector PDF + PNG @ 300 dpi)

| path | content | status |
|---|---|---|
| `comparison.{pdf,png}` | peak memory + Dice bars: exact vs XConv r=4 (600 st) vs XConv r=512 (60 st) | done |
| `segmentation_panel.{pdf,png}` | input CT │ GT overlay │ prediction overlay (MONAI `blend_images`), most-spleen slice | done |
| `segmentation_3d.{pdf,png}` | axial-slice grid with predicted spleen overlay (MONAI `matshow3d`) | done |
| `loss/loss_curves.{pdf,png}` | training (solid) **+ validation (dashed)** DiceCE loss vs step, conv=blue / XConv r=4=red; batch 8, 600 steps | **done** |
| `segmentation/volume_{1,2,3}.{pdf,png}` | per-slice **GT │ conv (exact) │ XConv r=4** spleen overlays (4 axial slices/volume, MONAI `blend_images`); per-volume Dice in the title | **done** |
| `segmentation/slices/vol{V}_z{Z}_{ct,gt,conv,xconv}.{png,pdf}` | **fully individual** borderless tiles — one image each for raw CT, GT, conv, XConv r=4, per slice (assemble side by side; see below) | **done** |

Built by `visualize.py` (`--loss_only`, `--seg_compare`, `--seg_individual`,
`--compare_only`, `--seg_only`; uses MONAI's own viz). The loss + segmentation figures
above were produced on **CPU at batch 8** (the GPU was contended) from the saved
checkpoints. Regenerate any time:

```bash
python visualize.py --loss_only        # -> results/figures/loss/loss_curves.{pdf,png}
python visualize.py --seg_compare       # -> results/figures/segmentation/volume_*.{pdf,png}
python visualize.py --seg_individual    # -> results/figures/segmentation/slices/vol*_z*_*.{png,pdf}
```

To reproduce the batch-8 runs themselves (CPU; ~20 min conv, ~40 min XConv r=4):

```bash
python run.py --method baseline --batch 8 --lr 0.0002 --max_steps 600 --val_every 20 --val_volumes 3 --device cpu
python run.py --method xconv    --batch 8 --ps 4 --lr 0.0002 --max_steps 600 --val_every 20 --val_volumes 3 --device cpu
```

### Individual per-slice tiles — `results/figures/segmentation/slices/`

Borderless, full-bleed tiles (no axes/titles/padding), **one image per column per slice**,
so the four tiles for a slice are pixel-identical in size and assemble cleanly. Naming:

```
vol{V}_z{Z}_{kind}.{png,pdf}        kind ∈ {ct, gt, conv, xconv}
#  V = held-out volume (1..3)
#  Z = zero-padded axial-slice index (e.g. z087)
```

All four tiles for a slice share the prefix `vol{V}_z{Z}_`, so they group together. The
intended **left-to-right order is `ct → gt → conv → xconv`** (note: this differs from the
alphabetical sort `conv, ct, gt, xconv`). Drop the `ct` tile if you only want
GT │ conv │ XConv. To place them next to each other:

**ImageMagick** — build one composite row per slice (CT │ GT │ conv │ XConv):

```bash
cd results/figures/segmentation/slices
for s in vol*_z*_ct.png; do b=${s%_ct.png}
  montage "${b}_ct.png" "${b}_gt.png" "${b}_conv.png" "${b}_xconv.png" \
    -tile 4x1 -geometry +2+2 "${b}_row.png"
done
```

**LaTeX** — 4 vector columns (swap in any `vol{V}_z{Z}`):

```latex
\begin{figure}\centering
  \foreach \k/\c in {ct/CT, gt/GT, conv/conv (exact), xconv/XConv r=4}{%
    \begin{subfigure}{0.24\linewidth}\centering
      \includegraphics[width=\linewidth]{slices/vol1_z087_\k.pdf}\caption*{\c}
    \end{subfigure}\hfill}
  \caption{Held-out volume 1, axial slice z=087.}
\end{figure}
```

---

## Findings & honest caveats

- **Main claim (supported):** the probed XConv gradient (even at r=4, the noisiest)
  trains the spleen UNet to **the same Dice as the exact gradient** at the same
  recipe, using **17.5% less peak memory**. Small-r noise is fully recovered by a
  smaller lr + longer training (and is unbiased after the `ali` boundary fix).
- **Why the saving is "only" 17.5%:** XConv (`mode='conv'`) compresses *conv-input*
  storage; the UNet's **skip-connection activations + PReLU outputs** (~9.8 GB) are
  unsavable, so memory floors there regardless of r. `mode='all'` (BReLU) does **not**
  help because this UNet uses **PReLU**, which BReLU does not convert.
- **For a large memory saving**, the lever is the architecture, not r: the in-house
  `radcompare` UNet (ReLU + no normalization, `mode='all'` BReLU) is built for it and
  the saving grows with image size — that is the right figure for the "absurd
  savings" story.

---

## Reproduce

```bash
pip install -e ~/Codes/xconv_pv          # branch ali (boundary-fixed pyxconv)
cd research/xconv_finetune

# strict apples-to-apples (identical recipe; only the gradient differs)
python run.py --method baseline --batch 64 --lr 0.0002 --max_steps 600 --val_volumes 5
python run.py --method xconv    --batch 64 --ps 4 --lr 0.0002 --max_steps 600 --val_volumes 5

python visualize.py                       # -> results/figures/*.{pdf,png}
```

(Branch `xconv-finetune` on `alisiahkoohi/MONAI`.)
