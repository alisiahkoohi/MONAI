"""Where does XConv save memory? Sweep GPU batch and compare peak memory of
baseline (exact Conv3d) vs XConv at a fixed r.

Peak memory is ``radcompare.memory.peak_memory_mib`` (``torch_peak``, 2-iter
warm-up) -- the same metric as the rest of the experiment. XConv's footprint is
nearly batch-independent (the probe e=(spatial x r) does not grow with batch), so
the baseline's linear growth should cross above XConv. Synthetic inputs (real
cuDNN ops) so it is fast and apples-to-apples.
"""
from __future__ import annotations

import argparse
import os

import torch
from radcompare import peak_memory_mib

import bundle as B
import xconv_ops as xc


def _peak_or_oom(make_model, gpu_batch, loss_fn, device):
    try:
        model = make_model()
        x, y = B.synthetic_batch(gpu_batch, device)
        peak = peak_memory_mib(model, x, y, loss_fn=loss_fn)
        del model, x, y
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        peak = None
    torch.cuda.empty_cache()
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=int, default=256)
    p.add_argument("--batches", default="8,16,32,64,96,128")
    p.add_argument("--dataset_dir", default=os.path.join(os.path.dirname(__file__),
                                                         "data", "Task09_Spleen"))
    a = p.parse_args()
    dd = a.dataset_dir
    device = torch.device("cuda")
    loss_fn = B.loss_fn(dd)
    torch.backends.cudnn.benchmark = True
    print(f"peak memory (MiB) vs GPU batch | XConv r={a.r}, patch 96, UNet\n")
    print(f"{'batch':>6} | {'baseline':>10} | {'xconv':>10} | winner")
    print("-" * 44)
    for bsz in [int(v) for v in a.batches.split(",")]:
        base = _peak_or_oom(lambda: B.bare_net(dd, device), bsz, loss_fn, device)
        xv = _peak_or_oom(
            lambda: xc.apply_xconv(B.bare_net(dd, device), a.r, "independent", "conv"),
            bsz, loss_fn, device)
        bs = "OOM" if base is None else f"{base:.0f}"
        xs = "OOM" if xv is None else f"{xv:.0f}"
        if base is None and xv is not None:
            win = "xconv (baseline OOM)"
        elif base is not None and xv is not None:
            win = "xconv" if xv < base else "baseline"
        else:
            win = "-"
        print(f"{bsz:>6} | {bs:>10} | {xs:>10} | {win}")


if __name__ == "__main__":
    main()
