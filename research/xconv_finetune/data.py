"""Spleen (MSD Task09) data pipeline and a synthetic batch for memory probing.

Standard MONAI CT-spleen preprocessing: RAS orientation, 1.5x1.5x2.0 mm spacing,
abdominal-window intensity scaling, foreground crop, then a single positive/
negative-balanced random cubic patch per volume so the DataLoader's ``batch_size``
is the literal training batch.

``synthetic_batch`` returns tensors with the exact shapes a train step consumes,
used by the capacity probes in ``memory.py`` (no disk/network, no data bias).
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from monai.apps import DecathlonDataset
from monai.data import list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, EnsureTyped,
)

KEYS = ("image", "label")
IN_CHANNELS = 1
OUT_CHANNELS = 2  # background + spleen


def _train_transform(patch: int) -> Compose:
    return Compose([
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes="RAS"),
        Spacingd(keys=KEYS, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=-57.0, a_max=164.0,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=KEYS, source_key="image"),
        RandCropByPosNegLabeld(
            keys=KEYS, label_key="label", spatial_size=(patch, patch, patch),
            pos=1, neg=1, num_samples=1, image_key="image", image_threshold=0.0,
        ),
        EnsureTyped(keys=KEYS),
    ])


def _val_transform() -> Compose:
    return Compose([
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes="RAS"),
        Spacingd(keys=KEYS, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=-57.0, a_max=164.0,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=KEYS, source_key="image"),
        EnsureTyped(keys=KEYS),
    ])


def train_loader(data_dir: str, patch: int, batch_size: int,
                 cache_num: int, num_workers: int) -> DataLoader:
    ds = DecathlonDataset(
        root_dir=data_dir, task="Task09_Spleen", section="training",
        transform=_train_transform(patch), download=True,
        cache_num=cache_num, num_workers=num_workers,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, collate_fn=list_data_collate,
                      drop_last=True, pin_memory=True)


def val_dataset(data_dir: str, num_workers: int) -> DecathlonDataset:
    return DecathlonDataset(
        root_dir=data_dir, task="Task09_Spleen", section="validation",
        transform=_val_transform(), download=True,
        cache_num=0, num_workers=num_workers,
    )


def synthetic_batch(batch: int, patch: int, device: torch.device
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Image/label tensors matching a real train step, for memory probing."""
    img = torch.randn(batch, IN_CHANNELS, patch, patch, patch, device=device)
    lbl = torch.randint(0, OUT_CHANNELS, (batch, 1, patch, patch, patch),
                        device=device, dtype=torch.float32)
    return img, lbl
