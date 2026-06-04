"""Where does XConv actually save memory? Sweep GPU batch and compare the TRUE
(NVML) peak of baseline vs XConv at a fixed r. XConv's footprint is nearly
batch-independent (the probe e=(spatial x r) does not grow with batch), so the
baseline's linear growth should cross above XConv and OOM first.

Synthetic inputs (real cuDNN ops) so the sweep is fast and apples-to-apples.
"""
from __future__ import annotations

import argparse
import os
import threading
import time

import torch
import pynvml
from monai.optimizers import Novograd

import bundle as B
import engine
import xconv_ops as xc

pynvml.nvmlInit()
_H = pynvml.nvmlDeviceGetHandleByIndex(0)
_mb = lambda: pynvml.nvmlDeviceGetMemoryInfo(_H).used / 1024**2


class NvmlPeak:
    def __init__(self, interval=0.004): self.interval = interval
    def __enter__(self):
        self.peak = _mb(); self._stop = False
        self._t = threading.Thread(target=self._poll, daemon=True); self._t.start(); return self
    def _poll(self):
        while not self._stop:
            self.peak = max(self.peak, _mb()); time.sleep(self.interval)
    def __exit__(self, *a):
        self._stop = True; self._t.join(); torch.cuda.synchronize()
        self.peak = max(self.peak, _mb())


def peak_or_oom(make_model, gpu_batch, loss_fn, lr, device, steps=3, warmup=2):
    try:
        model = make_model()
        opt = Novograd(model.parameters(), lr=lr)
        img, lbl = B.synthetic_batch(gpu_batch, device)
        for _ in range(warmup):
            engine.train_step(model, img, lbl, loss_fn, opt)
        with NvmlPeak() as pk:
            for _ in range(steps):
                engine.train_step(model, img, lbl, loss_fn, opt)
        out = pk.peak
    except RuntimeError as e:
        out = None if "out of memory" in str(e).lower() else (_ for _ in ()).throw(e)
    finally:
        for n in ("opt", "model", "img", "lbl"):
            if n in dir():
                pass
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=int, default=256)
    p.add_argument("--batches", default="8,16,32,64,96,128,192")
    a = p.parse_args()
    dd = os.path.join(os.path.dirname(__file__), "data", "Task09_Spleen")
    device = torch.device("cuda")
    loss_fn = B.loss_fn(dd); lr = B.optimizer_lr(dd)
    torch.backends.cudnn.benchmark = True
    print(f"NVML peak (MB) vs GPU batch | XConv r={a.r}, patch 96, UNet\n")
    print(f"{'batch':>6} | {'baseline':>10} | {'xconv':>10} | winner")
    print("-" * 44)
    for bsz in [int(x) for x in a.batches.split(",")]:
        base = peak_or_oom(lambda: B.bare_net(dd, device), bsz, loss_fn, lr, device)
        xv = peak_or_oom(lambda: xc.apply_xconv(B.bare_net(dd, device), a.r, "independent", "conv"),
                         bsz, loss_fn, lr, device)
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
