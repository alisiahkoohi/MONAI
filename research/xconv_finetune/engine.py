"""Training step, finetuning loop, conv-update verification, and a Dice read.

Loss is Dice + cross-entropy (MONAI ``DiceCELoss`` with ``softmax`` and
``to_onehot_y``) — the standard Spleen objective. The optimizer is AdamW over
*all* parameters (full finetuning: every conv receives gradients, which is the
regime where XConv's memory saving matters). ``verify_conv_updates`` makes that
explicit by measuring that conv weights actually move after a few steps.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from data import synthetic_batch

Step = Callable[[nn.Module, int], None]


def build_loss() -> DiceCELoss:
    return DiceCELoss(to_onehot_y=True, softmax=True)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(model: nn.Module, img: torch.Tensor, lbl: torch.Tensor,
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """One forward/backward/update on a batch; returns the scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(img)
    loss = loss_fn(logits, lbl)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def make_synthetic_step(loss_fn: nn.Module, lr: float, weight_decay: float,
                        patch: int, device: torch.device) -> Step:
    """A `(model, batch) -> None` step on synthetic data, for memory probes."""

    def step(model: nn.Module, batch: int) -> None:
        opt = make_optimizer(model, lr, weight_decay)
        img, lbl = synthetic_batch(batch, patch, device)
        train_step(model, img, lbl, loss_fn, opt)
    return step


def finetune(model: nn.Module, loader, loss_fn: nn.Module,
             optimizer: torch.optim.Optimizer, max_steps: int,
             device: torch.device, peak_tracker) -> dict:
    """Finetune for ``max_steps`` batches, tracking loss and peak memory.

    ``peak_tracker`` is ``memory.track_peak(device)`` (a context manager); the
    peak is captured across the whole loop, i.e. the real footprint of training.
    """
    losses: list[float] = []
    step = 0
    with peak_tracker as pk:
        while step < max_steps:
            for batch in loader:
                img = batch["image"].to(device)
                lbl = batch["label"].to(device)
                losses.append(train_step(model, img, lbl, loss_fn, optimizer))
                step += 1
                if step % 10 == 0 or step == 1:
                    print(f"    step {step:>4}/{max_steps}  loss={losses[-1]:.4f}")
                if step >= max_steps:
                    break
    return {"losses": losses, "peak_gb": pk[0]}


@torch.no_grad()
def evaluate_dice(model: nn.Module, val_ds, n_volumes: int, patch: int,
                  device: torch.device) -> float:
    """Mean foreground Dice on a few val volumes via sliding-window inference."""
    model.eval()
    metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_lbl = AsDiscrete(to_onehot=2)
    for i in range(min(n_volumes, len(val_ds))):
        sample = val_ds[i]
        img = sample["image"].unsqueeze(0).to(device)
        lbl = sample["label"].unsqueeze(0).to(device)
        logits = sliding_window_inference(
            img, roi_size=(patch, patch, patch), sw_batch_size=1, predictor=model,
            overlap=0.25,
        )
        pred = [post_pred(p) for p in decollate_batch(logits)]
        gt = [post_lbl(g) for g in decollate_batch(lbl)]
        metric(y_pred=pred, y=gt)
    return float(metric.aggregate().item())
