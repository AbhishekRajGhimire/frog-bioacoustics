"""Training loop with early stopping, and batch prediction."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .augment import augment_batch
from .config import TrainingConfig
from .model import SpectrogramResNet, to_model_input


PREDICTION_BATCH_SIZE = 256


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_average_precision: float
    learning_rate: float
    seconds: float


@dataclass(frozen=True)
class FitResult:
    model: SpectrogramResNet
    best_epoch: int
    history: tuple[EpochRecord, ...]


def select_device(requested: str | None = None) -> torch.device:
    """The requested device, or the GPU when one is available and the CPU otherwise."""
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    return device


def use_full_precision() -> None:
    """Keep GPU arithmetic in full fp32 so GPU and CPU predictions agree.

    TF32 is faster but keeps only about three significant digits, which would
    make a saved model's predictions depend on the hardware that runs it.
    """
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"


def fit_model(
    train_pixels: np.ndarray,
    train_labels: np.ndarray,
    val_pixels: np.ndarray,
    val_labels: np.ndarray,
    config: TrainingConfig,
    *,
    db_floor: float,
    db_ceiling: float,
    device: torch.device,
    progress: Callable[[str], None] = print,
) -> FitResult:
    """Train from the configured starting weights and return the best validation epoch's model."""
    if len(train_pixels) < 2:
        raise ValueError("training needs at least two images")
    _seed_everything(config.seed)
    use_full_precision()
    settings = config.optimization
    model = SpectrogramResNet(pretrained=config.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.max_epochs)
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(config.seed)
    pixels = torch.as_tensor(train_pixels)
    targets = torch.as_tensor(train_labels, dtype=torch.float32)
    val_targets = torch.as_tensor(val_labels, dtype=torch.float32)

    history: list[EpochRecord] = []
    best_state = copy.deepcopy(model.state_dict())
    best_precision = -math.inf
    best_epoch = 0
    for epoch in range(1, settings.max_epochs + 1):
        started = time.perf_counter()
        learning_rate = scheduler.get_last_lr()[0]
        model.train()
        seen, total_loss = 0, 0.0
        for index in _batches(len(pixels), settings.batch_size, generator):
            batch = augment_batch(
                to_model_input(pixels[index]),
                config.augmentation,
                db_floor=db_floor,
                db_ceiling=db_ceiling,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch.to(device)), targets[index].to(device))
            loss.backward()
            optimizer.step()
            seen += len(index)
            total_loss += loss.item() * len(index)
        scheduler.step()

        val_logits = predict_logits(model, val_pixels, device)
        val_loss = float(loss_function(torch.from_numpy(val_logits), val_targets))
        val_precision = float(average_precision_score(val_labels, val_logits))
        record = EpochRecord(epoch, total_loss / seen, val_loss, val_precision, learning_rate, time.perf_counter() - started)
        history.append(record)
        improved = val_precision > best_precision
        if improved:
            best_precision, best_epoch = val_precision, epoch
            best_state = copy.deepcopy(model.state_dict())
        progress(
            f"epoch {epoch:>2}  train loss {record.train_loss:.4f}  val loss {val_loss:.4f}  "
            f"val AP {val_precision:.4f}  {record.seconds:.1f}s{'  best' if improved else ''}"
        )
        if epoch - best_epoch >= settings.early_stopping_patience:
            progress(f"stopping: no validation improvement for {settings.early_stopping_patience} epochs")
            break

    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model=model, best_epoch=best_epoch, history=tuple(history))


def predict_logits(model: nn.Module, pixels: np.ndarray, device: torch.device) -> np.ndarray:
    """One logit per image, without augmentation, in input order."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(pixels), PREDICTION_BATCH_SIZE):
            batch = to_model_input(pixels[start : start + PREDICTION_BATCH_SIZE]).to(device)
            outputs.append(model(batch).float().cpu())
    return torch.cat(outputs).numpy()


def predict_probabilities(model: nn.Module, pixels: np.ndarray, device: torch.device) -> np.ndarray:
    """Frog probability per image, in input order."""
    logits = torch.from_numpy(predict_logits(model, pixels, device)).double()
    return torch.sigmoid(logits).numpy()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batches(count: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    """Shuffled index batches. A final partial batch is dropped when other batches
    exist, because BatchNorm statistics and the gradient from a handful of images
    are noisy; reshuffling every epoch means no image is left out for good."""
    batches = list(torch.split(torch.randperm(count, generator=generator), batch_size))
    if len(batches) > 1 and len(batches[-1]) < batch_size:
        batches.pop()
    return batches
