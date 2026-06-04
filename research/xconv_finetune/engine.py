"""Training step and finetuning loop (pure: no data/model/bundle coupling).

The loss/optimizer/dataloader are supplied by the caller (here, all from the
MONAI bundle). Peak memory is captured across the whole loop via ``peak_tracker``
(``memory.track_peak``), i.e. the real footprint of finetuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def train_step(model: nn.Module, img: torch.Tensor, lbl: torch.Tensor,
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """One forward/backward/update on a batch; returns the scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(img), lbl)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def finetune(model: nn.Module, loader, loss_fn: nn.Module,
             optimizer: torch.optim.Optimizer, max_steps: int,
             device: torch.device, peak_tracker, scheduler=None) -> dict:
    """Finetune for ``max_steps`` batches, tracking loss and peak memory.

    ``scheduler`` (e.g. the bundle's StepLR) is stepped per iteration, matching
    the bundle's per-step schedule.
    """
    losses: list[float] = []
    step = 0
    with peak_tracker as pk:
        while step < max_steps:
            for batch in loader:
                img = batch["image"].to(device)
                lbl = batch["label"].to(device)
                losses.append(train_step(model, img, lbl, loss_fn, optimizer))
                if scheduler is not None:
                    scheduler.step()
                step += 1
                if step % 10 == 0 or step == 1:
                    print(f"    step {step:>4}/{max_steps}  loss={losses[-1]:.4f}")
                if step >= max_steps:
                    break
    return {"losses": losses, "peak_gb": pk[0]}
