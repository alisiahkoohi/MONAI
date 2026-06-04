"""True GPU peak memory via NVML (CUDA context + torch reserved + cuDNN workspace).

torch.cuda.max_memory_allocated only counts live tensor allocations and misses the
cuDNN convolution workspace -- exactly the memory XConv changes. We poll NVML in a
background thread during the measured region and take the max.
"""
from __future__ import annotations

import threading
import time

import torch
import pynvml

pynvml.nvmlInit()
_H = pynvml.nvmlDeviceGetHandleByIndex(0)


def nvml_used_gb() -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(_H).used / 1024**3


class NvmlPeak:
    """Context manager: poll true GPU usage; ``.peak_gb`` holds the max seen."""
    def __init__(self, interval: float = 0.004):
        self.interval = interval
        self.peak_gb = 0.0

    def __enter__(self):
        self.peak_gb = nvml_used_gb()
        self._stop = False
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def _poll(self):
        while not self._stop:
            u = nvml_used_gb()
            if u > self.peak_gb:
                self.peak_gb = u
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop = True
        self._t.join()
        torch.cuda.synchronize()
        self.peak_gb = max(self.peak_gb, nvml_used_gb())
