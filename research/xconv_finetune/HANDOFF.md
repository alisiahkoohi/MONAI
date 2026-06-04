# HANDOFF — XConv memory comparison on the MONAI spleen UNet

For the next Claude. Read this fully before touching anything. It captures the
goal, the standing research contract, the current state, every hard-won finding,
and the open items.

---

## 0. TL;DR

We finetune the **pretrained MONAI `spleen_ct_segmentation` UNet** twice — once
with **regular Conv3d** (baseline, repo recipe) and once with **XConv** (probed
low-memory weight gradient) — and compare **true GPU peak memory (NVML)**. The
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

`pyxconv` is **NOT installed yet**. Before any run: `pip install -e ~/Codes/xconv_pv`
(branch `ali`; add `--no-deps` to avoid touching torch). Editable so the live
boundary-fixed tree is used. The fork's code does a plain `import pyxconv`.

---

## 3. The experiment (what to run, once GPU is freed)

Repo recipe (all from the bundle's `configs/train.json`): UNet, **Novograd
lr=0.002**, **StepLR(5000,0.1)**, **DiceCELoss**, GPU batch = 4 crops × loader_bs.

**The two comparisons (see README for exact commands):**
1. `python run.py --method baseline --batch 64 --max_steps 60` → records
   `nvml_peak_gb` (the ceiling).
2. `python calibrate.py --batch 64 --rs 128,256,384,512` → largest `r` with NVML
   peak ≤ baseline; then `python run.py --method xconv --batch 64 --ps <R>
   --max_steps 60`.

Memory is **true NVML** (`gpu_mem.NvmlPeak`), not `torch.cuda.max_memory_allocated`.

### File map
`bundle.py` (load pretrained UNet/loss/lr/scheduler/loader) · `xconv_ops.py`
(Conv3d→Xconv3D + init-preservation guard) · `gpu_mem.py` (NVML peak poller) ·
`memory.py` (torch-allocator sizing: `find_max_ps`, `maximize_ps`) · `engine.py`
(train step + loop) · `run.py` (driver) · `calibrate.py` (max r ≤ baseline NVML) ·
`sweep_batch.py` (baseline-vs-XConv NVML vs batch — finds the crossover).

---

## 4. Critical findings (the journey — do not relearn these the hard way)

1. **Measure memory with NVML, not the torch allocator.**
   `torch.cuda.max_memory_allocated` misses the **cuDNN conv-workspace** and CUDA
   context — exactly the region XConv changes. Use `gpu_mem.NvmlPeak` (polls
   `nvmlDeviceGetMemoryInfo().used`). The user's tracker is
   `~/Codes/xconv_pv/pyxconv/nvidia_mem_tracker.py`.
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

1. **`pip install -e ~/Codes/xconv_pv` (branch `ali`)** before any run — pending.
2. **Two-agent Gate-1 review of the PIVOTED (bundle) code** — the review was done
   on the earlier SegResNet/self-pretrain version, which was then deleted. The
   current bundle-based code has **not** been re-reviewed. Per the contract, do
   this before promoting results.
3. **Run the two comparisons** (baseline + xconv at batch 64) once GPU is free and
   reviewed. Confirm XConv `nvml_peak_gb` ≤ baseline.
4. **`run.py`'s XConv auto-sizing uses torch peak + 15 GB card budget**
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
