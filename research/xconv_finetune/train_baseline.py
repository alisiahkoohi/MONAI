"""Finetune a 3D SegResNet on Spleen with regular Conv3d (the baseline).

Picks the largest batch that fits the GPU budget, finetunes all layers, verifies
the conv weights actually update, and records peak memory.

Usage:
    python train_baseline.py                 # auto batch, patch 96, 60 steps
    python train_baseline.py --patch 96 --max_steps 100 --batch_size 2
"""
from __future__ import annotations

from config import parse_args
from finetune import run

if __name__ == "__main__":
    run(parse_args(method="baseline"))
