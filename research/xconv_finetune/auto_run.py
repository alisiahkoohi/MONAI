"""Autonomous sizing + baseline/XConv comparison, under the canonical metric.

Picks the operating point with ``radcompare.peak_memory_mib`` (torch_peak, 2-iter):
the largest GPU batch whose **XConv** peak stays <= the **baseline** peak, and the
largest probing count ``r`` at that batch with peak <= baseline. Then runs baseline
and XConv back to back via ``run.py`` (bundle repo recipe).

IMPORTANT: the parent process does NO GPU work. Sizing runs in its OWN subprocess
(``auto_run.py --size``) so all its GPU memory is released on exit before the runs
launch -- otherwise the parent's reserved memory OOMs the run.py children.

    python auto_run.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

BATCHES = [32, 48, 64, 80, 96, 112, 128]      # ascending, multiples of NUM_SAMPLES=4
R_LADDER = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
STEPS = "30"
MIN_FREE_MIB = 12000
ENV = dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True")


# ----------------------------------------------------------------------------- size
def do_size():
    """GPU sizing; prints sizing lines + a final 'RESULT batch=.. r=..' to stdout."""
    import torch
    from radcompare import peak_memory_mib
    import bundle as B
    import xconv_ops as xc

    free = torch.cuda.mem_get_info()[0] / 1024**2
    if free < MIN_FREE_MIB:
        print(f"[size] GPU not free ({free:.0f} MiB) -> abort"); print("RESULT none reason=busy")
        return
    torch.backends.cudnn.benchmark = True
    loss_fn = B.loss_fn(os.path.join(_HERE, "data", "Task09_Spleen"))
    dd = os.path.join(_HERE, "data", "Task09_Spleen")

    def peak(make_model, batch):
        try:
            m = make_model()
            x, y = B.synthetic_batch(batch, torch.device("cuda"))
            p = peak_memory_mib(m, x, y, loss_fn=loss_fn)
            del m, x, y
            torch.cuda.empty_cache()
            return p
        except RuntimeError as e:
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                return None
            raise

    print(f"[size] GPU free ({free:.0f} MiB); sizing (XConv peak <= baseline) ...", flush=True)
    candidates = []
    for b in BATCHES:
        base = peak(lambda: B.bare_net(dd, torch.device("cuda")), b)
        if base is None:
            print(f"[size] batch {b}: baseline OOM -> stop", flush=True); break
        best_r = 0
        for r in R_LADDER:
            xp = peak(lambda: xc.apply_xconv(B.bare_net(dd, torch.device("cuda")), r,
                                             "independent", "conv"), b)
            if xp is None:
                break
            if xp <= base:
                best_r = r
            else:
                break
        print(f"[size] batch {b}: baseline {base:.0f} MiB | max r<=baseline = {best_r}", flush=True)
        if best_r > 0:
            candidates.append((b, base, best_r))
    if candidates:
        b, base, r = candidates[-1]
        print(f"RESULT batch={b} r={r} base={base:.0f}", flush=True)
    else:
        print("RESULT none reason=no_crossover", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    # sizing in its OWN subprocess: GPU memory fully released on exit.
    print("[auto] sizing (subprocess) ...", flush=True)
    out = subprocess.run([sys.executable, __file__, "--size"], cwd=_HERE, env=ENV,
                         capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    if out.returncode != 0:
        print(f"[auto] sizing subprocess failed:\n{out.stderr[-800:]}", flush=True); sys.exit(1)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), "")
    if "none" in line or not line:
        print(f"[auto] no feasible (batch,r): {line or 'no RESULT'} -> not running", flush=True)
        sys.exit(2)
    kv = dict(tok.split("=") for tok in line.split()[1:])
    b, r = kv["batch"], kv["r"]
    print(f"[auto] chosen: batch={b}, r={r} (baseline {kv.get('base','?')} MiB)", flush=True)

    for cmd in (["--method", "baseline", "--batch", b, "--max_steps", STEPS],
                ["--method", "xconv", "--batch", b, "--ps", r, "--max_steps", STEPS]):
        print(f"[auto] run.py {' '.join(cmd)}", flush=True)
        subprocess.run([sys.executable, "run.py"] + cmd, cwd=_HERE, env=ENV, check=True)

    res = os.path.join(_HERE, "results")
    try:
        bp = json.load(open(os.path.join(res, f"baseline_unet_spleen_b{b}.json")))["peak_mib"]
        xp = json.load(open(os.path.join(res,
              f"xconv_unet_spleen_b{b}_r{r}_independent_conv.json")))["peak_mib"]
        print(f"[auto] DONE  batch={b} r={r} | baseline {bp:.0f} MiB vs XConv {xp:.0f} MiB "
              f"({'XConv <= baseline OK' if xp <= bp else 'XConv > baseline'})", flush=True)
    except Exception as e:
        print(f"[auto] DONE (results read failed: {e})", flush=True)


if __name__ == "__main__":
    if "--size" in sys.argv:
        do_size()
    else:
        main()
