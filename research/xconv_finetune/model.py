"""SegResNet construction, XConv conversion, and conv-layer introspection.

SegResNet (MONAI) is a 3D ResNet-style encoder/decoder built from regular
``nn.Conv3d`` (groups=1) and GroupNorm; its default upsampling is parameter-free
interpolation, so there are no transposed convs to worry about. Regular convs are
exactly the case XConv's probed weight-gradient supports, so ``convert_net``
swaps every ``Conv3d`` for an ``Xconv3D`` in place (same weights, low-memory
backward).

This module also exposes the tools used to *prove finetuning trains the convs*:
list the conv layers, snapshot their weights, and measure the per-layer update
after optimizer steps.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

# vendored, patched pyxconv (no nvidia_smi/matplotlib import) lives beside this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyxconv  # noqa: E402


def build_segresnet(in_channels: int, out_channels: int, init_filters: int,
                    seed: int, device: torch.device) -> nn.Module:
    """Construct a SegResNet with explicit, local determinism."""
    torch.manual_seed(seed)
    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )
    return model.to(device)


def load_pretrained(model: nn.Module, ckpt_path: str) -> dict:
    """Load a SegResNet ``state_dict`` in place; tolerate output-head mismatch.

    Keys whose shapes disagree (e.g. a different number of output classes from a
    cross-task checkpoint) are skipped and the report lists them, so the body
    transfers and only the incompatible head stays freshly initialized. This must
    run on the *plain* SegResNet, BEFORE any XConv conversion.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    own = model.state_dict()
    keep, skip = {}, []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            skip.append(k)
    missing = [k for k in own if k not in keep]
    model.load_state_dict(keep, strict=False)
    return {"loaded": len(keep), "skipped_shape_mismatch": skip,
            "not_in_checkpoint": [k for k in missing if k not in skip]}


def apply_xconv(model: nn.Module, ps: int, xmode: str, target: str) -> nn.Module:
    """Replace every ``Conv3d`` (and optionally ReLU) by its XConv counterpart.

    ``target='conv'`` converts only convolutions (the controlled comparison);
    ``target='all'`` additionally swaps ReLU for the memory-saving ``BReLU``.
    ``ps`` is the number of probing vectors (memory vs. gradient-noise knob).

    Conversion REUSES each conv's existing weight Parameter (it does not
    re-initialize), so pretrained weights are preserved; ``assert_convert_preserved``
    checks this invariant after the fact.
    """
    if ps <= 0:
        raise ValueError(f"xconv requires ps > 0, got {ps}")
    pyxconv.convert_net(model, ps=ps, xmode=xmode, mode=target)
    return model


@torch.no_grad()
def assert_convert_preserved(model: nn.Module,
                             before: dict[str, torch.Tensor]) -> float:
    """Raise if XConv conversion changed any conv weight; return the max delta.

    Guards the user-critical invariant that ``convert_net`` does not destroy the
    (pretrained) initialization. ``before`` is ``snapshot_conv_weights`` taken
    immediately before ``apply_xconv``.
    """
    deltas = conv_update_deltas(model, before)
    if not deltas:
        return 0.0
    worst = max(deltas.values())
    if worst > 0.0:
        offenders = [n for n, d in deltas.items() if d > 0][:5]
        raise RuntimeError(
            f"convert_net altered conv weights (max rel. delta {worst:.3e}); "
            f"pretrained init not preserved. First offenders: {offenders}")
    return worst


# --- conv-layer introspection / update verification ----------------------------

def conv_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """All Conv3d-family layers (``Xconv3D`` subclasses ``nn.Conv3d``)."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv3d)]


def count_xconv(model: nn.Module) -> int:
    return sum(isinstance(m, pyxconv.Xconv3D) for _, m in model.named_modules())


def conv_trainable_report(model: nn.Module) -> dict:
    """Summarize that conv weights carry gradients (i.e. are being finetuned)."""
    convs = conv_layers(model)
    trainable = sum(m.weight.requires_grad for _, m in convs)
    n_params = sum(m.weight.numel() for _, m in convs)
    return {
        "n_conv_layers": len(convs),
        "n_conv_trainable": trainable,
        "n_conv_weight_params": int(n_params),
        "n_xconv_layers": count_xconv(model),
    }


@torch.no_grad()
def snapshot_conv_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: m.weight.detach().clone() for n, m in conv_layers(model)}


@torch.no_grad()
def conv_update_deltas(model: nn.Module,
                       before: dict[str, torch.Tensor]) -> dict[str, float]:
    """Relative L2 change ``||W_after - W_before|| / ||W_before||`` per conv."""
    out = {}
    for n, m in conv_layers(model):
        w0 = before[n]
        denom = w0.norm().item() + 1e-12
        out[n] = (m.weight.detach() - w0).norm().item() / denom
    return out
