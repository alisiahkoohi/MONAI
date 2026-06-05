# HANDOFF — XConv memory comparison on the MONAI spleen UNet

For the next Claude. Read this fully before touching anything. It captures the
goal, the standing research contract, the current state, every hard-won finding,
and the open items.

---

## 0. TL;DR

We finetune the **pretrained MONAI `spleen_ct_segmentation` UNet** twice — once
with **regular Conv3d** (baseline, repo recipe) and once with **XConv** (probed
low-memory weight gradient) — and compare **peak GPU memory** (the rad-vs-xconv
metric `radcompare.peak_memory_mib` = `torch_peak`, 2-iter warm-up). The
point of XConv is to *not use more memory than the baseline* while training the
conv layers. Code is in this directory; it is committed on branch
`xconv-finetune` and pushed to the user's fork **`alisiahkoohi/MONAI`**
(commit `0070a9f`).

**Nothing has been run on the GPU for the final comparison yet** — the user
keeps the GPU paused and gates each step. Do not launch GPU work without an
explicit "go".

---

## 1. The standing contract (from `~/Codes/new_repo/claude_research_starter.md`)

That file is the **binding contract** for all research work in `~/Codes/` for Ali
Siahkoohi. Read it in full at the start. The essentials:

**Coding house style** (refs: `~/Codes/flatvi/` = cleanest template,
`~/Codes/icnnlift_luqi/` = at scale; conventions in `~/Codes/new_repo/README.md`):
- **Super-factorized, minimum spaghetti, `projorg`-driven, reproducible.**
- One concept per file (<~250–360 lines); **models are inert, the trainer
  drives**; dependency injection via a `builders.py`; split "loss" from
  "minimizer"; depend on structural interfaces, not concrete classes.
- `projorg` is **mandatory** for real experiments (auto-named runs, checkpoint/
  plot/log dirs, git-hash logging): `setup_environment`, `checkpointsdir`,
  `plotsdir`, `logsdir`, `upload_to_cloud`. Config-as-identity. Never hand-roll
  experiment names / argparse loops / upload scripts.
- Reuse the **`sips`** conda env; do not invent a new one.
- Local determinism (`torch.manual_seed` right before model construction); no
  global seed_everything; `print()`+`tqdm`, no `logging`.

**The two-agent over-the-shoulder review loop (CORE WORKFLOW — do not skip):**
After writing a unit of code, spawn **two agents in parallel** —
(1) **coding/scalability** agent, (2) **math** agent — to review the finished
code (and any math-bearing writeup). Report both agents' feedback to the user,
fix, report the fixes. **Gate 1 (code):** start experiments only after **the user
AND both agents** are satisfied. **Gate 2 (results):** promote results to a
writeup only after both agents judge the actual figures/logs satisfactory.

**Writing (when asked for a writeup/report):** staged, with an **editor agent**
enforcing style (`~/Codes/new_repo/recent_style.md` primary, `BARANIUK_STYLE.md`,
`STYLE_GUIDE_5PAPERS.md`). Phase 1 outline → editor pass → fix; Phase 2
paragraph-by-paragraph → editor pass → fix. **Figures publication-grade** (vector
PDF, viridis, gold star, despined, semantic palette). **Captions and results
prose state NO numbers** — qualitative, in Ali's voice; numbers are read off the
figure.

**Deviation in effect for THIS task:** the user explicitly chose to **fork MONAI
and work in a branch** (option "Fork Project-MONAI/MONAI") rather than scaffold a
`projorg` repo. So the `projorg`/`new_repo.sh` layout is **waived here**; the code
is plain factorized modules under `research/xconv_finetune/`. The two-agent review
loop and gates still apply (see Open Items — the *pivoted* code has NOT been
re-reviewed).

---

## 2. Where everything lives

| Thing | Path / value |
|---|---|
| This experiment | `~/Codes/MONAI/research/xconv_finetune/` (branch `xconv-finetune`) |
| Pushed fork | `alisiahkoohi/MONAI`, branch `xconv-finetune`, commit `0070a9f` |
| In-house XConv (USE THIS) | `~/Codes/xconv_pv`, branch **`ali`** — boundary-fixed (2D+3D), **uncommitted working tree** |
| Python env | `sips`: `/home/al289197/miniconda3/envs/sips/bin/python` (torch 2.9+cu128, MONAI 1.5.2, pynvml, timm, nibabel, einops) |
| GPU | NVIDIA RTX 2000 Ada, **16 GB** (shared — another agent may use it; user gates GPU) |
| Spleen data (downloaded) | `research/xconv_finetune/data/Task09_Spleen/` (gitignored) |
| Bundle (downloaded) | `research/xconv_finetune/bundles/spleen_ct_segmentation/` (gitignored) |

`pyxconv` is **installed editable** (`pip install -e ~/Codes/xconv_pv --no-deps`),
pointing at the live boundary-fixed `ali` tree; `find_packages()` means
**`radcompare` is importable too**. The fork does a plain `import pyxconv`. (Watch
for a stale `pyxconv/__pycache__` in the fork dir shadowing it as an empty
namespace package — delete any `research/xconv_finetune/pyxconv/` that reappears.)

---

## 3. The experiment (what to run, once GPU is freed)

Repo recipe (all from the bundle's `configs/train.json`): UNet, **Novograd
lr=0.002**, **StepLR(5000,0.1)**, **DiceCELoss**, GPU batch = 4 crops × loader_bs.

**The two comparisons (see README for exact commands):**
1. `python run.py --method baseline --batch 64 --max_steps 60` → records
   `peak_mib` (the ceiling).
2. `python calibrate.py --batch 64 --rs 128,256,384,512` → largest `r` with peak
   ≤ baseline; then `python run.py --method xconv --batch 64 --ps <R>
   --max_steps 60`.

Memory is the **single canonical metric everywhere**: `radcompare.peak_memory_mib`
(`torch_peak` from the repo `MemoryTracker`, 2-iteration warm-up, SGD lr=0).

### File map
`bundle.py` (load pretrained UNet/loss/lr/scheduler/loader) · `xconv_ops.py`
(Conv3d→Xconv3D + init-preservation guard) · `memory.py` (sizing via
`peak_memory_mib`: `find_max_ps`, `maximize_ps`) · `engine.py` (train step + loop) ·
`run.py` (driver, reports `peak_mib`) · `calibrate.py` (max r whose peak ≤ baseline) ·
`sweep_batch.py` (baseline-vs-XConv peak vs batch — finds the crossover) ·
`age_memory.py` (AGE + peak-memory vs r — the rad-vs-xconv figures).

### AGE + peak-memory figures (rad-vs-xconv methodology)
`age_memory.py` produces the two canonical figures, **reusing `radcompare`
verbatim** so numbers match the paper:
- **AGE** (paper Eq. 9, `radcompare.age.average_gradient_error` /
  `exact_full_gradient`): exact full-dataset gradient vs minibatch gradient on the
  conv weights. Exact model's AGE = sampling floor; XConv decays toward it as `r`
  grows (unbiased after the `ali` boundary fix — no plateau).
- **Peak memory** (`radcompare.memory.peak_memory_mib`): **`torch_peak` via the
  repo `MemoryTracker`, 2-iteration warm-up** (i=0 warms cuDNN, i=1 measured).
- **Their `max_batch_for_budget`** (binary search for the largest batch fitting a
  fixed memory budget) is the elegant framing: XConv's lower memory buys a larger
  batch → lower AGE. (Their UNet sweep is `run_unet_age_vs_imgdim` over *image
  dimension*; ours sweeps `r` at native 96³.)
- Run: `python age_memory.py --rs 2,4,...,256 --batch 8 --subset 64 --n_runs 3`.
  CPU-validated end-to-end with `--synthetic --skip_memory` (AGE decays with r).

**Metric: unified (was an inconsistency, now fixed).** Every script measures peak
via `radcompare.peak_memory_mib` (`torch_peak`, 2-iter, SGD lr=0) — the paper's
reported metric. The earlier NVML poller (`gpu_mem.py`) was removed. Note this
metric excludes the real optimizer's state (Novograd buffers) by design (matches
the paper); the live finetune holds a bit more, but the *reported* comparison is
the canonical number. If you ever want the device-level figure too, the `ali`
`MemoryTracker` also exposes `nvidia_used`.

---

## 4. Critical findings (the journey — do not relearn these the hard way)

1. **Use the rad-vs-xconv canonical peak metric — `radcompare.peak_memory_mib`.**
   It is `torch_peak` from the repo `MemoryTracker` with a 2-iteration warm-up and
   SGD lr=0 — the number the paper reports. (NVML/`nvidia_used` is also exposed by
   the tracker and includes context+cuDNN workspace, but `torch_peak` is the
   methodology to match; an earlier NVML poller was removed for consistency.)
2. **cuDNN autotuning is a transient spike.** Cold first step (benchmark=True)
   spikes ~2 GB then settles. Always **warm up, then measure** for fair
   baseline-vs-XConv comparison (calibrate/sweep do this).
3. **XConv only beats *exact-conv* memory with BReLU.** With ReLU left exact
   (`--xconv_target conv`), XConv only *matches* exact memory; the paper's BReLU
   (`mode='all'`) compresses ReLU activations into a real saving. **BUT the MONAI
   spleen UNet uses PReLU**, which `BReLU` does not convert — so `mode='all'`
   does **not** help here. (The in-house `scripts/rad_vs_xconv_unet.py` uses a
   ReLU Ronneberger UNet on purpose.)
4. **XConv's footprint is ~batch-independent; the win appears at large batch.**
   The probe `e=(spatial×r)` does not grow with batch; baseline grows linearly.
   `sweep_batch.py` showed the **crossover at batch ~64** (XConv 13.1 vs baseline
   14.3 GB); **both OOM at batch 96** because the **UNet skip connections** hold
   high-res activations XConv cannot compress. So on this UNet the memory win is
   modest (~8%) and batch-capped.
5. **`r` (probing count) trades memory vs gradient noise; high-res 3D layers are
   expensive.** The first conv at 96³ makes `e=(96³ × r)` huge, so `r` is gated
   by the highest-resolution converted layer, not by total activations. Dropping
   batch does NOT raise the `r` ceiling.
6. **`convert_net` preserves the pretrained init** — it reuses each conv's weight
   `Parameter` (verified: `max|ΔW|=0`). The code loads pretrained **then**
   converts and hard-asserts preservation (`assert_convert_preserved`).
7. **Boundary-bias fix** (the user's `ali` branch): the probe's flat `e.roll`
   estimated the *circular*-conv gradient, biasing it ~20–45% for k>1 vs the
   zero-padded conv. `ali` is padding-aware for **2D and 3D** (verified upstream
   by `verify_fix2.py`). Memory results don't depend on it; gradient fidelity
   does. (My earlier standalone `check_xconv_grad.py` independently found this and
   was removed as redundant.)

---

## 5. Open items / next steps

1. **Editable install — DONE** (`pip install -e ~/Codes/xconv_pv --no-deps`;
   verified fixed `ali` tree + `radcompare` importable). CPU prep done:
   `age_memory.py` written and CPU-smoke-validated (`--synthetic --skip_memory`).
2. **Two-agent Gate-1 review of the PIVOTED (bundle) code + `age_memory.py`** — the
   review was done on the earlier SegResNet/self-pretrain version, which was then
   deleted. The current bundle-based code has **not** been re-reviewed. Per the
   contract, do this before promoting results.
3. **Run, once GPU is free + reviewed:** (a) the two finetuning comparisons
   (`run.py` baseline + xconv at batch 64) — confirm XConv ≤ baseline; (b) the
   AGE + peak-memory figures (`age_memory.py`, full `r` sweep on GPU).
3b. **Memory metric — DONE/unified:** every script now uses
   `radcompare.peak_memory_mib`; the NVML poller was removed. (Open sub-point: the
   metric excludes the live Novograd optimizer state by design — fine for the paper
   comparison, but note it if you report absolute training footprint.)
4. **`run.py`'s XConv auto-sizing uses `peak_memory_mib` + 15 GB card budget**
   (`maximize_ps`), NOT the NVML ≤baseline constraint. The README's procedure
   sidesteps this by using `calibrate.py` + explicit `--ps`. Consider rewiring
   `run.py` to size against the baseline's NVML peak directly.
5. **Decision the user is weighing:** the spleen UNet (PReLU, skip connections)
   is a weak showcase for XConv memory. Options surfaced: accept the modest
   batch-64 result; switch to a ReLU net + `mode='all'` (cf. their
   `rad_vs_xconv_unet.py`); or a non-skip architecture. Confirm direction.
6. **Writeup**, if requested: phased, editor-agent-reviewed, publication-grade
   figures, **no numbers in prose/captions** (contract §4–5).

---

## 6. Gotchas / etiquette specific to this user

- The user **gates the GPU** and pauses often ("pause gpu", "don't run anything").
  Never launch GPU work without an explicit go. CPU setup (pip, file edits, git)
  is usually fine but confirm if unsure.
- **Use the in-house `pyxconv` (`ali`), never stock slimgroup/XConv** — only the
  in-house one has the boundary fix and (2D) grouped/depthwise support. It is
  **PyTorch-only and the 3D weight-grad path is dense-only** (groups=1 required —
  fine for this UNet).
- Another agent may be editing `~/Codes/xconv_pv`. Do not commit its working tree
  unless asked; just `pip install -e` reads it live.
- Background long jobs (`run_in_background: true`) and read the `.output` file;
  python buffers stdout, so output appears at the end.
- `gh` is authenticated as `alisiahkoohi`. The fork `alisiahkoohi/MONAI` exists;
  push the branch with `git push fork xconv-finetune` (remote `fork`).
- Commit selectively ("only necessary scripts"); `bundles/`, `data/`, `results/`
  are gitignored.
