"""Finetune a 3D SegResNet on Spleen with XConv convs (low-memory backward).

Reuses the baseline's batch (sized identically), then picks the largest probing
count ``ps`` that still fits the GPU budget at that batch, so the activation
memory XConv frees is spent on a less-noisy probed gradient. Finetunes all
layers, verifies conv updates, and records peak memory.

Usage:
    python train_xconv.py                    # auto batch, auto ps, independent mode
    python train_xconv.py --ps 1024 --xmode gaussian --xconv_target all
"""
from __future__ import annotations

from config import parse_args
from finetune import run

if __name__ == "__main__":
    run(parse_args(method="xconv"))
