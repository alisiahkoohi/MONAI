"""Largest probing count r whose TRUE GPU peak stays <= the baseline's.

Memory is the real NVML device usage (CUDA context + torch reserved + cuDNN
workspace) -- not torch.cuda.max_memory_allocated, which misses the cuDNN conv
workspace that XConv avoids. We poll NVML during warmed-up finetune steps and take
the max, for the baseline and for XConv at a ladder of r (same batch, same loader).
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


def _nvml_used_mb() -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(_H).used / 1024**2


class NvmlPeak:
    """Poll true GPU memory in a thread; report the max seen (MB)."""
    def __init__(self, interval=0.004):
        self.interval = interval
    def __enter__(self):
        self.peak = _nvml_used_mb(); self._stop = False
        self._t = threading.Thread(target=self._poll, daemon=True); self._t.start()
        return self
    def _poll(self):
        while not self._stop:
            self.peak = max(self.peak, _nvml_used_mb())
            time.sleep(self.interval)
    def __exit__(self, *a):
        self._stop = True; self._t.join()
        torch.cuda.synchronize()
        self.peak = max(self.peak, _nvml_used_mb())


def true_peak_mb(model, loader, loss_fn, lr, device, steps=4, warmup=2) -> tuple[float, float]:
    """(nvml_peak_mb, torch_peak_mb) over measured steps, after warmup (autotune)."""
    opt = Novograd(model.parameters(), lr=lr)
    it = iter(loader)
    def one():
        b = next(it)
        engine.train_step(model, b["image"].to(device), b["label"].to(device), loss_fn, opt)
    for _ in range(warmup):
        one()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with NvmlPeak() as pk:
        for _ in range(steps):
            one()
    tp = torch.cuda.max_memory_allocated() / 1024**2
    del opt, it
    torch.cuda.empty_cache()
    return pk.peak, tp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--rs", default="64,128,256,512,1024")
    p.add_argument("--target", default="conv")
    a = p.parse_args()

    dd = os.path.join(os.path.dirname(__file__), "data", "Task09_Spleen")
    device = torch.device("cuda")
    loss_fn = B.loss_fn(dd); lr = B.optimizer_lr(dd)
    batch = a.batch or B.repo_batch(dd)
    loader = B.train_loader(dd, batch, a.num_workers)
    torch.backends.cudnn.benchmark = True

    base = B.pretrained_net(dd, device)
    base_nvml, base_torch = true_peak_mb(base, loader, loss_fn, lr, device, a.steps)
    del base; torch.cuda.empty_cache()
    print(f"\n[baseline] batch={batch}: NVML peak {base_nvml:6.0f} MB | torch peak {base_torch:6.0f} MB"
          f"  (ceiling = NVML {base_nvml:.0f} MB)\n")

    best = 0
    for r in [int(x) for x in a.rs.split(",")]:
        net = xc.apply_xconv(B.pretrained_net(dd, device), r, "independent", a.target)
        nv, tp = true_peak_mb(net, loader, loss_fn, lr, device, a.steps)
        ok = nv <= base_nvml
        print(f"[xconv r={r:<5}] NVML peak {nv:6.0f} MB | torch peak {tp:6.0f} MB  "
              f"{'<= baseline OK' if ok else '> baseline'}")
        del net; torch.cuda.empty_cache()
        if ok:
            best = r
        else:
            break
    print(f"\n==> largest r with NVML peak <= baseline ({base_nvml:.0f} MB) at batch {batch}: r = {best}")


if __name__ == "__main__":
    main()
