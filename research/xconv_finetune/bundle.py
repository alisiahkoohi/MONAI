"""Thin wrapper over the MONAI ``spleen_ct_segmentation`` bundle.

We do NOT pretrain and we do NOT reinvent the pipeline: the bundle provides the
pretrained UNet (``models/model.pt``), the loss (``DiceCELoss``), and the data
pipeline (``configs/train.json``). We only finetune FROM the bundle's pretrained
weights, with or without XConv, and measure memory.

Batching note: the bundle crop emits ``NUM_SAMPLES`` patches per volume and the
DataLoader batches ``loader_bs`` volumes, so the GPU batch is
``B = NUM_SAMPLES * loader_bs`` (and B >= NUM_SAMPLES keeps the UNet's BatchNorm
well-defined).
"""
from __future__ import annotations

import os

import torch
from monai.bundle import ConfigParser, download

_HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(_HERE, "bundles")
BUNDLE = "spleen_ct_segmentation"
NUM_SAMPLES = 4          # RandCropByPosNegLabeld num_samples -> B = NUM_SAMPLES * loader_bs
IN_CHANNELS = 1
OUT_CHANNELS = 2
PATCH = 96               # bundle crop side


def ensure_bundle() -> str:
    """Path to the bundle, downloading it on first use."""
    root = os.path.join(BUNDLE_DIR, BUNDLE)
    if not os.path.exists(os.path.join(root, "models", "model.pt")):
        os.makedirs(BUNDLE_DIR, exist_ok=True)
        download(name=BUNDLE, bundle_dir=BUNDLE_DIR)
    return root


def _parser(dataset_dir: str) -> ConfigParser:
    root = ensure_bundle()
    p = ConfigParser()
    p.read_config(os.path.join(root, "configs", "train.json"))
    p["bundle_root"] = root
    p["dataset_dir"] = os.path.abspath(dataset_dir)
    return p


def bare_net(dataset_dir: str, device: torch.device) -> torch.nn.Module:
    """Bundle UNet WITHOUT pretrained weights — for memory sizing (value-independent)."""
    return _parser(dataset_dir).get_parsed_content("network").to(device)


def pretrained_net(dataset_dir: str, device: torch.device) -> torch.nn.Module:
    """Bundle UNet with the pretrained spleen weights loaded (the finetuning init)."""
    parser = _parser(dataset_dir)
    net = parser.get_parsed_content("network").to(device)
    sd = torch.load(os.path.join(ensure_bundle(), "models", "model.pt"),
                    map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[bundle] UNet pretrained load: missing={len(missing)} unexpected={len(unexpected)} "
          f"(0/0 = exact)")
    return net


def loss_fn(dataset_dir: str):
    return _parser(dataset_dir).get_parsed_content("loss")


def optimizer_lr(dataset_dir: str) -> float:
    """The bundle optimizer's learning rate (Novograd, lr=0.002), used verbatim."""
    return float(_parser(dataset_dir).get_parsed_content("optimizer#lr"))


def repo_batch(dataset_dir: str) -> int:
    """The bundle's own GPU batch = NUM_SAMPLES crops * the loader batch_size."""
    return NUM_SAMPLES * int(_parser(dataset_dir)["train#dataloader#batch_size"])


def make_scheduler(optimizer, dataset_dir: str):
    """Rebuild the bundle's LR scheduler (StepLR(5000, 0.1)) on our optimizer."""
    import importlib
    cfg = dict(_parser(dataset_dir)["lr_scheduler"])
    target = cfg.pop("_target_")
    cfg.pop("optimizer", None)
    mod, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)(optimizer, **cfg)


def train_loader(dataset_dir: str, gpu_batch: int, num_workers: int):
    """A bundle train DataLoader whose GPU batch is ``gpu_batch`` (multiple of NUM_SAMPLES)."""
    if gpu_batch % NUM_SAMPLES != 0:
        raise ValueError(f"gpu_batch must be a multiple of NUM_SAMPLES={NUM_SAMPLES}, "
                         f"got {gpu_batch}")
    parser = _parser(dataset_dir)
    parser["train#dataloader#batch_size"] = gpu_batch // NUM_SAMPLES
    parser["train#dataloader#num_workers"] = num_workers
    return parser.get_parsed_content("train#dataloader")


def synthetic_batch(gpu_batch: int, device: torch.device):
    """Image/label tensors matching a real step, for the memory-sizing probes."""
    img = torch.randn(gpu_batch, IN_CHANNELS, PATCH, PATCH, PATCH, device=device)
    lbl = torch.randint(0, OUT_CHANNELS, (gpu_batch, 1, PATCH, PATCH, PATCH),
                        device=device, dtype=torch.float32)
    return img, lbl
