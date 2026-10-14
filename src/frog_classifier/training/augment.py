"""Seeded augmentation of training batches.

Images are float tensors shaped (batch, 1, mels, frames) with values in [0, 1],
where 0 is the display floor (the recording's noise floor under the current
contract) and 1 is the display ceiling.
"""

from __future__ import annotations

import torch

from .config import AugmentationConfig


def augment_batch(
    images: torch.Tensor,
    config: AugmentationConfig,
    *,
    db_floor: float,
    db_ceiling: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return an augmented copy. Draws come from the generator in a fixed order."""
    augmented = _time_shift(images, generator)
    augmented = _attenuate(augmented, config.max_attenuation_db, db_floor, db_ceiling, generator)
    augmented = _mask(augmented, config.time_mask_max_frames, dim=3, generator=generator)
    return _mask(augmented, config.frequency_mask_max_bins, dim=2, generator=generator)


def _time_shift(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Roll every image circularly along time, because a call can start anywhere in a window."""
    batch, _, _, frames = images.shape
    offsets = torch.randint(0, frames, (batch,), generator=generator)
    columns = (torch.arange(frames).unsqueeze(0) - offsets.unsqueeze(1)) % frames
    return torch.gather(images, 3, columns[:, None, None, :].expand_as(images))


def _attenuate(
    images: torch.Tensor,
    max_db: float,
    db_floor: float,
    db_ceiling: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Make each image quieter by up to max_db, as if its sources were farther away.

    Pixels store decibels above the noise floor. The energy above the floor is
    scaled down in the power domain, so the floor itself stays where it is.
    """
    if max_db == 0:
        return images
    db_range = db_ceiling - db_floor
    attenuation_db = torch.rand(images.shape[0], generator=generator) * max_db
    gain = torch.pow(10.0, -attenuation_db / 10.0).view(-1, 1, 1, 1)
    level_db = db_floor + images * db_range
    excess = torch.pow(10.0, level_db / 10.0) - 1.0
    quieter_db = 10.0 * torch.log10(1.0 + excess * gain)
    return ((quieter_db - db_floor) / db_range).clamp(0.0, 1.0)


def _mask(images: torch.Tensor, max_width: int, *, dim: int, generator: torch.Generator) -> torch.Tensor:
    """Set one random block of up to max_width positions along dim to the floor."""
    if max_width == 0:
        return images
    batch, size = images.shape[0], images.shape[dim]
    widths = torch.randint(0, min(max_width, size) + 1, (batch,), generator=generator)
    starts = (torch.rand(batch, generator=generator) * (size - widths + 1)).floor().long()
    positions = torch.arange(size)
    masked = (positions >= starts[:, None]) & (positions < (starts + widths)[:, None])
    shape = [batch, 1, 1, 1]
    shape[dim] = size
    return images.masked_fill(masked.view(shape), 0.0)
