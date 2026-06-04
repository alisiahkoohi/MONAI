"""Experiment configuration for XConv finetuning of a 3D SegResNet on Spleen.

One flat namespace of hyperparameters, parsed from the CLI. Two thin drivers
(``train_baseline.py``, ``train_xconv.py``) preset ``method`` and otherwise share
this config and the orchestration in ``finetune.py``.

The memory study fixes a single ``batch_size`` for both methods (the value that
fits the GPU for the *baseline*), then lets XConv spend its freed activation
memory on more probing vectors ``ps`` at that same batch.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    method: str = "baseline"           # "baseline" (regular Conv3d) or "xconv"
    # data
    data_dir: str = os.path.join(_HERE, "data")
    patch: int = 96                    # cubic crop side, voxels
    cache_num: int = 8                 # volumes cached in RAM (keep modest)
    num_workers: int = 4
    # model
    init_filters: int = 16             # SegResNet width
    # finetuning
    batch_size: int = 0                # 0 -> auto-pick largest that fits the budget
    max_steps: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_volumes: int = 2               # held-out volumes for a final Dice read
    # pretraining / initialization
    pretrained_ckpt: str = ""          # state_dict to finetune FROM ("" -> random init)
    pretrain_steps: int = 200          # steps for the pretrain.py phase that makes one
    # xconv
    ps: int = 0                        # 0 -> auto-pick largest "r" that fits the budget
    xmode: str = "independent"         # probe mode: independent | gaussian | orthogonal
    xconv_target: str = "conv"         # convert "conv" only (clean) or "all" (+BReLU)
    size_strategy: str = "max_ps"      # "max_ps": maximize r, drop batch as needed;
                                       # "fixed_batch": keep the baseline batch, max r at it
    # system / bookkeeping
    mem_budget_gb: float = 15.0        # stay under the 16 GB card
    seed: int = 0
    device: str = "cuda"
    out_dir: str = os.path.join(_HERE, "results")
    tag: str = ""                      # optional suffix for the run name

    def run_name(self) -> str:
        base = f"{self.method}_segresnet_spleen_p{self.patch}_b{self.batch_size}"
        if self.method == "xconv":
            base += f"_ps{self.ps}_{self.xmode}_{self.xconv_target}"
        return base + (f"_{self.tag}" if self.tag else "")

    def as_dict(self) -> dict:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    """Every Config field becomes a typed ``--flag`` with its default."""
    p = argparse.ArgumentParser(description=__doc__)
    for f, default in Config().as_dict().items():
        if isinstance(default, bool):
            p.add_argument(f"--{f}", type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=default)
        else:
            p.add_argument(f"--{f}", type=type(default), default=default)
    return p


def parse_args(method: str | None = None, argv: list[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    cfg = Config(**vars(args))
    if method is not None:
        cfg.method = method
    if cfg.method not in ("baseline", "xconv"):
        raise ValueError(f"method must be 'baseline' or 'xconv', got {cfg.method!r}")
    if cfg.xmode not in ("independent", "gaussian", "orthogonal"):
        raise ValueError(f"xmode must be independent|gaussian|orthogonal, got {cfg.xmode!r}")
    if cfg.xconv_target not in ("conv", "all"):
        raise ValueError(f"xconv_target must be conv|all, got {cfg.xconv_target!r}")
    if cfg.size_strategy not in ("max_ps", "fixed_batch"):
        raise ValueError(f"size_strategy must be max_ps|fixed_batch, got {cfg.size_strategy!r}")
    return cfg
