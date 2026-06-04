"""GPU peak-memory measurement and capacity sizing.

The whole point of XConv is to lower the activation memory of the backward pass.
We therefore measure *peak allocated* memory across a full train step (the moment
that decides whether a config fits), and provide two capacity probes used to set
the experiment's knobs:

* ``find_max_batch`` — largest batch that fits the budget for the *baseline*.
* ``find_max_ps``    — largest XConv probing count that fits at a *fixed* batch.

Each probe rebuilds the model and runs one real forward/backward/step on
synthetic tensors (a failed step can leave CUDA state dirty, so we never reuse a
model across attempts).
"""
from __future__ import annotations

import contextlib
import gc
from typing import Callable

import torch


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1e9


@contextlib.contextmanager
def track_peak(device: torch.device):
    """Context manager yielding a one-element list that holds the peak GB."""
    reset_peak(device)
    out: list[float] = [0.0]
    try:
        yield out
    finally:
        out[0] = peak_gb(device)


def _is_oom(err: RuntimeError) -> bool:
    return "out of memory" in str(err).lower()


def _free_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def one_step_peak(
    build_model: Callable[[], torch.nn.Module],
    step_fn: Callable[[torch.nn.Module, int], None],
    batch: int,
    device: torch.device,
) -> float | None:
    """Run a (warmed-up) train step; return peak GB, or ``None`` on OOM.

    ``build_model`` returns a fresh model on ``device``; ``step_fn(model, batch)``
    runs one forward/backward/optimizer step on synthetic data. A first,
    *unmeasured* step is run to trigger cuDNN autotuning/workspace allocation, so
    the measured peak reflects what the real loop will actually allocate at this
    shape (otherwise the probe under-counts and the chosen knob can OOM).
    """
    model = build_model()
    try:
        step_fn(model, batch)               # warmup: autotune + allocate workspace
        with track_peak(device) as pk:
            step_fn(model, batch)           # measured
        return pk[0]
    except RuntimeError as err:
        if _is_oom(err):
            return None
        raise
    finally:
        del model
        gc.collect()                        # drop a failed attempt's frame refs
        _free_cuda(device)


def find_max_batch(
    build_model: Callable[[], torch.nn.Module],
    step_fn: Callable[[torch.nn.Module, int], None],
    device: torch.device,
    budget_gb: float,
    candidates: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
    margin: float = 0.9,
) -> tuple[int, float]:
    """Largest candidate batch whose peak stays under ``margin*budget_gb``.

    The ``margin`` leaves headroom for fragmentation and any workspace the probe
    does not reproduce, so the chosen batch does not OOM the real loop.
    """
    cap = margin * budget_gb
    best, best_mem = 0, 0.0
    for b in candidates:
        mem = one_step_peak(build_model, step_fn, b, device)
        fits = mem is not None and mem <= cap
        print(f"  [batch probe] b={b:<3} peak={'OOM' if mem is None else f'{mem:5.2f} GB'}"
              f"  (cap {cap:.2f}) -> {'fits' if fits else 'reject'}")
        if not fits:
            break
        best, best_mem = b, mem
    if best == 0:
        raise RuntimeError("Even batch=1 does not fit the memory budget.")
    return best, best_mem


def find_max_ps(
    build_converted_model: Callable[[int], torch.nn.Module],
    step_fn: Callable[[torch.nn.Module, int], None],
    batch: int,
    device: torch.device,
    budget_gb: float,
    candidates: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
    margin: float = 0.9,
) -> tuple[int, float]:
    """Largest XConv ``ps`` whose peak stays under ``margin*budget_gb`` at ``batch``.

    ``build_converted_model(ps)`` returns a fresh XConv-converted model on device.
    """
    cap = margin * budget_gb
    best, best_mem = 0, 0.0
    for ps in candidates:
        mem = one_step_peak(lambda: build_converted_model(ps), step_fn, batch, device)
        fits = mem is not None and mem <= cap
        print(f"  [ps probe] ps={ps:<5} peak={'OOM' if mem is None else f'{mem:5.2f} GB'}"
              f"  (cap {cap:.2f}) -> {'fits' if fits else 'reject'}")
        if not fits:
            break
        best, best_mem = ps, mem
    if best == 0:
        raise RuntimeError("No probing count fits the memory budget at this batch.")
    return best, best_mem


# probing-count ladder reaching far past 2D ranges — 3D + small batch can afford a lot
_PS_LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def maximize_ps(
    build_converted_model: Callable[[int], torch.nn.Module],
    step_fn: Callable[[torch.nn.Module, int], None],
    device: torch.device,
    budget_gb: float,
    max_batch: int,
    batch_ladder: tuple[int, ...] = (16, 12, 8, 6, 4, 3, 2, 1),
    min_batch: int = 1,
    margin: float = 0.9,
) -> tuple[int, int, float]:
    """Maximize the probing count ``ps`` (``r``), dropping batch only as needed.

    Smaller batch frees memory for a larger ``ps``, so the maximum ``ps`` is found
    by scanning batches from ``max_batch`` down to 1 and taking the largest ``ps``
    that fits. Because ``ps`` is non-decreasing as batch shrinks, the first
    (largest) batch that reaches the best ``ps`` wins — i.e. the batch is dropped
    only as far as a strictly larger ``ps`` requires.

    Returns ``(ps, batch, peak_gb)``.
    """
    batches = sorted({b for b in batch_ladder if min_batch <= b <= max_batch} | {min_batch},
                     reverse=True)
    best_ps, best_b, best_mem = 0, 0, 0.0
    for b in batches:
        try:
            ps, mem = find_max_ps(build_converted_model, step_fn, b, device,
                                  budget_gb, _PS_LADDER, margin)
        except RuntimeError:
            ps, mem = 0, 0.0       # not even the smallest ps fits at this batch
        print(f"  [max-r] batch={b:<3} -> best ps={ps:<6} (peak {mem:5.2f} GB)")
        if ps > best_ps:
            best_ps, best_b, best_mem = ps, b, mem
    if best_ps == 0:
        raise RuntimeError("No (batch, ps) combination fits the memory budget.")
    return best_ps, best_b, best_mem
