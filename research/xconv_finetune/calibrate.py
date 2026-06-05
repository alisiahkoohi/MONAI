"""Largest probing count r whose peak memory stays <= the baseline's.

Peak memory is the rad-vs-xconv canonical metric: ``radcompare.memory.peak_memory_mib``
(``torch_peak`` from the repo ``MemoryTracker``, 2-iteration warm-up, SGD lr=0 so the
optimizer state is not counted). The SAME function is used in ``age_memory.py``,
``sweep_batch.py``, and (the reported number in) ``run.py`` -- one metric everywhere,
consistent with the in-house methodology.
"""
from __future__ import annotations

import argparse
import os

import torch
from radcompare import peak_memory_mib

import bundle as B
import xconv_ops as xc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=0)       # GPU batch (default: repo batch)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--rs", default="64,128,256,512,1024")
    p.add_argument("--target", default="conv")
    p.add_argument("--dataset_dir", default=os.path.join(os.path.dirname(__file__),
                                                         "data", "Task09_Spleen"))
    a = p.parse_args()

    device = torch.device("cuda")
    loss_fn = B.loss_fn(a.dataset_dir)
    batch = a.batch or B.repo_batch(a.dataset_dir)
    x, y = B.synthetic_batch(batch, device)              # values irrelevant to memory
    torch.backends.cudnn.benchmark = True

    base = B.pretrained_net(a.dataset_dir, device)
    base_peak = peak_memory_mib(base, x, y, loss_fn=loss_fn)
    del base
    torch.cuda.empty_cache()
    print(f"\n[baseline] batch={batch}: peak {base_peak:7.0f} MiB  (ceiling)\n")

    best = 0
    for r in [int(v) for v in a.rs.split(",")]:
        net = xc.apply_xconv(B.pretrained_net(a.dataset_dir, device), r, "independent", a.target)
        peak = peak_memory_mib(net, x, y, loss_fn=loss_fn)
        ok = peak <= base_peak
        print(f"[xconv r={r:<5}] peak {peak:7.0f} MiB  {'<= baseline OK' if ok else '> baseline'}")
        del net
        torch.cuda.empty_cache()
        if ok:
            best = r
        else:
            break
    print(f"\n==> largest r with peak <= baseline ({base_peak:.0f} MiB) at batch {batch}: r = {best}")


if __name__ == "__main__":
    main()
