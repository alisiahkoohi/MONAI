# XConv finetuning of a 3D SegResNet on Spleen

Memory-vs-probing study for low-memory backpropagation (XConv) when **finetuning
the convolutional layers** of a 3D segmentation network.

- **Task**: MSD Task09 Spleen (CT, 1 channel, binary segmentation).
- **Model**: MONAI `SegResNet` (regular `Conv3d` throughout — the case XConv's
  probed weight-gradient supports).
- **Claim under test**: at a fixed batch size, replacing every `Conv3d` with
  `Xconv3D` lowers activation memory; the freed memory buys a larger probing
  count `ps` (less gradient noise) while still fitting the GPU.

## Layout (one concept per file)

| file | role |
|---|---|
| `config.py` | flat hyperparameter namespace + CLI |
| `data.py` | Spleen loaders, transforms, synthetic batch for probing |
| `model.py` | build SegResNet, apply XConv, conv-layer introspection/update checks |
| `memory.py` | peak-memory measurement, `find_max_batch`, `find_max_ps` |
| `engine.py` | loss, train step, finetune loop, conv-update verify, Dice eval |
| `finetune.py` | `run(cfg)` orchestration (size → finetune → verify → record) |
| `train_baseline.py` / `train_xconv.py` | thin drivers |
| `pyxconv/` | vendored probed-conv layers (`Xconv2D/3D`, `convert_net`) |

## Run (three steps)

```bash
cd research/xconv_finetune

# 0) pretrain a plain SegResNet on Spleen -> a checkpoint to finetune FROM
python pretrain.py --pretrain_steps 200 --patch 96      # -> results/pretrained_segresnet_p96_f16.pth
CKPT=results/pretrained_segresnet_p96_f16.pth

# 1) baseline finetune (regular Conv3d), batch B0 sized to fit 16 GB
python train_baseline.py --pretrained_ckpt $CKPT --max_steps 60

# 2) xconv finetune: maximize the probing count r (=ps), dropping batch as needed
python train_xconv.py    --pretrained_ckpt $CKPT --max_steps 60
```

Each run writes a JSON summary (resolved batch/`r`, peak GB, conv-update proof,
init-preservation delta, Dice) to `results/`; the loss curve goes to a `.npy`.

## Initialization is preserved through conversion (verified)

The user-critical invariant: **`convert_net` must not destroy the pretrained
init.** It does not — it reuses each conv's existing weight `Parameter`. The
pipeline enforces the safe order **build → load pretrained → convert**, then
`assert_convert_preserved` hard-checks every conv weight is identical after
conversion (`convert_preserved_max_delta == 0`). Only a cross-task output head
(shape mismatch) is skipped, and `load_pretrained` reports it.

## Sizing: maximize r, drop batch as needed (`size_strategy='max_ps'`, default)

The baseline fixes the largest batch `B0` that fits 16 GB. The XConv run then
**maximizes `r`** by scanning batches from `B0` down to 1, taking the largest `r`
that fits and keeping the *largest* batch that reaches it (batch drops only as far
as a strictly larger `r` requires). `--size_strategy fixed_batch` instead holds
`B0` and maximizes `r` at it.

## Notes / caveats

- **Finetuning trains the convs (magnitude-checked)**: `Δ>0` is not proof (AdamW
  weight decay moves weights with a zero gradient). The run reports
  `convs_gradient_trained` only when the min relative conv update exceeds the
  weight-decay-only drift (`lr*wd*steps`) by a wide margin.
- **`r` is the memory/gradient-noise knob, and 3D is demanding**:
  `check_xconv_grad.py` shows the forward is exact and the probed gradient is
  unbiased in direction, but at large 3D maps a modest `r` is very noisy — cos ~ 0
  at 48^3 with r=256. Early high-resolution layers need a large `r`; that is why
  we maximize it.
- **XConv is PyTorch-only and dense-conv-only here**: SegResNet convs are
  `groups=1` (required — the in-house 3D weight-grad probe is dense-only; 2D adds
  grouped/depthwise). Do not point `convert_net` at depthwise/separable nets.
- **`xmode='independent'`** gave the cleanest gradient direction (and is required
  for the grouped 2D path); `gaussian` is the upstream default.
- Default `xconv_target='conv'` converts only convolutions; `'all'` also swaps
  ReLU for memory-saving `BReLU`.
- Vendored `pyxconv/` is a snapshot of `luqigroup/xconv_pv`
  (`pyxconv/VENDORED_FROM.txt`); re-vendor for upstream changes.
