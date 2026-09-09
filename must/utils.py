"""Seeding, episode plumbing, metrics and checkpoint I/O."""

import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_task(task: Dict[str, torch.Tensor], device: torch.device):
    """Unwrap the DataLoader's batch dimension (one episode per batch)."""
    prepared = {}
    for key, value in task.items():
        if torch.is_tensor(value):
            prepared[key] = value[0].to(device, non_blocking=True)
        else:
            prepared[key] = value
    return prepared


def aggregate_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    return (logits.argmax(dim=-1) == labels.long()).float().mean()


def confidence_interval_95(values):
    """Mean and 95% confidence half-width over per-episode accuracies."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0
    mean = float(array.mean())
    confidence = float(1.96 * array.std() / np.sqrt(array.size))
    return mean, confidence


def print_and_log(log_file, message: str):
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def current_learning_rate(
    base_lr: float,
    iteration: int,
    total: int,
    warmup_iterations: int = 0,
):
    """Step schedule: full LR, then x0.5, x0.1, x0.01 at 30/50/70% of training.

    An optional linear warm-up runs first. Warm-up matters once LoRA and the
    zero-gated branches are on: they start from an exactly-frozen model, and a
    full-rate first step can knock the frame features away from the raw CLIP
    features that the frozen text query buffer was built against.
    """
    if warmup_iterations > 0 and iteration <= warmup_iterations:
        return base_lr * iteration / float(warmup_iterations)
    progress = iteration / max(total, 1)
    if progress >= 0.7:
        return base_lr * 0.01
    if progress >= 0.5:
        return base_lr * 0.1
    if progress >= 0.3:
        return base_lr * 0.5
    return base_lr


def atomic_torch_save(payload, path: str):
    """Write through a temporary file so an interrupted run cannot corrupt it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)
