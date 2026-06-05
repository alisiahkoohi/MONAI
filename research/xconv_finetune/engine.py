"""Training step and finetuning loop (pure: no data/model/bundle coupling).

The loss/optimizer/dataloader are supplied by the caller (here, all from the
MONAI bundle). Peak memory is measured separately by the caller via the canonical
``radcompare.peak_memory_mib`` (``torch_peak``, 2-iteration warm-up), so this loop
only trains and returns the loss curve.
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
             device: torch.device, scheduler=None) -> dict:
    """Finetune for ``max_steps`` batches, returning the loss curve.

    Peak memory is measured separately via the canonical ``peak_memory_mib`` so the
    number matches the rad-vs-xconv methodology. ``scheduler`` (the bundle's StepLR)
    is stepped per iteration.
    """
    losses: list[float] = []
    step = 0
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
    return {"losses": losses}
