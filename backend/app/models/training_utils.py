"""
ML Training Utilities

Helper functions for GPU training: mixed precision, gradient accumulation,
device detection, and checkpoint management.
"""

import os
from pathlib import Path
from typing import Optional

import torch

from app.utils.logging_config import logger


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_training_device() -> torch.device:
    """Detect and return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU detected)")

    return device


# ---------------------------------------------------------------------------
# Mixed Precision Context Manager
# ---------------------------------------------------------------------------

def get_mixed_precision_context(device: torch.device):
    """
    Return the appropriate autocast context for mixed precision training.

    Mixed precision uses float16 on GPU to speed up training and reduce memory.
    """
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    else:
        # No-op context manager for CPU
        return torch.amp.autocast(device_type="cpu", enabled=False)


def get_gradient_scaler(device: torch.device) -> torch.amp.GradScaler:
    """Create a GradScaler for mixed precision (only active on GPU)."""
    if device.type == "cuda":
        return torch.amp.GradScaler('cuda')
    else:
        return torch.amp.GradScaler('cpu', enabled=False)


# ---------------------------------------------------------------------------
# Checkpoint Save / Load
# ---------------------------------------------------------------------------

def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    additional_info: Optional[dict] = None,
):
    """
    Save a model checkpoint to disk.

    Args:
        model:           The PyTorch model.
        optimizer:       The optimizer.
        epoch:           Current epoch number.
        loss:            Current loss value.
        save_path:       File path to save the checkpoint.
        additional_info: Any extra metadata to store.
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path} (epoch {epoch}, loss {loss:.4f})")


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load a model checkpoint from disk.

    Returns:
        The full checkpoint dictionary (epoch, loss, etc.).
    """
    if device is None:
        device = get_training_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        f"Checkpoint loaded: {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f})"
    )

    return checkpoint
