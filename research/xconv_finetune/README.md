# XConv finetuning of the MONAI spleen UNet — memory comparison

Finetune the **pretrained** `spleen_ct_segmentation` MONAI bundle **with and
without XConv**, and compare **true GPU peak memory**. No pretraining (the init is
the bundle's `models/model.pt`) and no custom pipeline — the UNet, `DiceCELoss`,
`Novograd`, `StepLR`, and data all come from the bundle's `configs/train.json`.
XConv only swaps the convolutions for a probed, low-memory weight gradient.

## REQUIRED: use the boundary-fixed pyxconv (branch `ali`)

`import pyxconv` must resolve to the in-house package on the **`ali`** branch of
`~/Codes/xconv_pv`, which fixes the probe's boundary bias (the backward used a
circular `roll` that estimated the *circular*-conv gradient, biasing it ~20–45%
for k>1 vs the zero-padded conv). The `ali` branch is **padding-aware for both 2D
and 3D**. Install it **editable** so any further upstream fixes are picked up
automatically (do **not** vendor/copy — it goes stale):

```bash
pip install -e ~/Codes/xconv_pv          # branch ali; --no-deps if you must not touch torch
python -c "import pyxconv; print(pyxconv.__file__)"   # must point inside ~/Codes/xconv_pv
```

Env: the `sips` conda env (`/home/al289197/miniconda3/envs/sips`), torch 2.9 +
CUDA, MONAI 1.5.2, `pynvml`.

## The two comparisons (exact commands)

Run from this directory. Memory is the **true NVML** device peak (CUDA context +
reserved pool + cuDNN workspace), not `torch.cuda.max_memory_allocated`.

```bash
# 1) BASELINE — finetune the pretrained UNet with regular Conv3d, repo recipe
#    (Novograd lr=0.002, StepLR(5000,0.1), DiceCELoss). Records the NVML peak = ceiling.
python run.py --method baseline --batch 64 --max_steps 60
#    -> results/baseline_unet_spleen_b64.json   (field: nvml_peak_gb)

# 2) XCONV — same batch, probed convs. Pick the largest probing count r whose
#    NVML peak stays <= the baseline's, then finetune at that r.
python calibrate.py --batch 64 --rs 128,256,384,512        # -> largest r with NVML <= baseline
python run.py --method xconv --batch 64 --ps <R> --max_steps 60
#    -> results/xconv_unet_spleen_b64_r<R>_independent_conv.json
```

Both JSONs carry `nvml_peak_gb`, the conv-update proof, and (for XConv)
`convert_preserved_max_delta` (must be 0 — conversion preserves the pretrained
init). Compare the two `nvml_peak_gb`: **XConv must be ≤ baseline.**

### Choosing the batch

XConv's footprint is nearly **batch-independent** (the probe `e=(spatial×r)` does
not grow with batch) while the baseline grows ~linearly, so XConv only wins at
**larger batch**. `sweep_batch.py` locates the crossover:

```bash
python sweep_batch.py --r 256 --batches 8,16,32,64,96    # baseline vs xconv NVML peak per batch
```
On this UNet (patch 96, 16 GB card) the crossover is ~batch 64; both OOM by 96
(the UNet's skip connections pin the high-res activations, which XConv cannot
compress). Use batch 64 as the operating point above.

## Files (committed)

| file | role |
|---|---|
| `bundle.py` | load the bundle's pretrained UNet, loss, lr, scheduler, data loader |
| `xconv_ops.py` | swap Conv3d→Xconv3D, conv introspection, init-preservation guard |
| `gpu_mem.py` | true GPU peak via NVML polling (`NvmlPeak`) |
| `memory.py` | torch-allocator sizing helpers (`find_max_ps`, `maximize_ps`) |
| `engine.py` | train step + finetune loop (pure) |
| `run.py` | CLI: finetune a method, record NVML peak + conv-update + preservation |
| `calibrate.py` | largest `r` whose NVML peak ≤ baseline (the operating-point picker) |
| `sweep_batch.py` | baseline-vs-XConv NVML peak across batch (finds the crossover) |

## Notes / caveats

- **XConv beats *exact-conv* memory only with BReLU.** With ReLU left exact
  (`--xconv_target conv`, the default), XConv only *matches* exact-conv memory —
  the stored activations are pinned by the activation, not the conv. The in-house
  paper uses BReLU (`mode='all'`) to turn this into a saving. **The MONAI spleen
  UNet uses PReLU**, which `BReLU` does not convert, so `--xconv_target all` does
  not help here; expect XConv ≈ baseline (a win only past the batch crossover).
- **Boundary fix** lives in `pyxconv` (branch `ali`), 2D **and** 3D — verified
  upstream (`verify_fix2.py`). Memory results do not depend on it; gradient
  fidelity does.
- **`r` is the memory/gradient-noise knob**; `xmode='independent'` is the cleanest
  probe and the one used here.
- The bundle + Spleen data auto-download to `bundles/` and `data/` on first use.
