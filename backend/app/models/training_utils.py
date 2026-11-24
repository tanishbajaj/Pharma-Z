"""
ML Training Utilities

Helper functions for GPU training: mixed precision, gradient accumulation,
device detection, and checkpoint management.
"""

import os
from pathlib import Path
from typing import Optional, Sequence

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
    allow_missing: Sequence[str] = (),
) -> dict:
    """
    Load a model checkpoint from disk.

    The load is strict by default: if the checkpoint's keys do not match the
    model, this raises rather than quietly leaving layers at random init. A
    model that silently runs on random weights produces plausible-looking
    garbage and reports nothing wrong, which is worse than failing.

    Args:
        allow_missing: Key prefixes the caller knowingly handles — e.g. a head
            added to the architecture after this checkpoint was trained. Keys
            under these prefixes may be absent from the checkpoint and are left
            at their initial values; every other mismatch still raises. The
            caller is responsible for making the model behave sensibly without
            those weights.

    Returns:
        The full checkpoint dictionary (epoch, loss, etc.).

    Raises:
        RuntimeError: on any key or shape mismatch not covered by
            `allow_missing`.
    """
    if device is None:
        device = get_training_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # strict=False relaxes only missing/unexpected keys — shape mismatches
    # still raise here — so we police the key sets ourselves below.
    incompatible = model.load_state_dict(state_dict, strict=False)

    allowed = tuple(allow_missing)
    expected_missing = [k for k in incompatible.missing_keys if k.startswith(allowed)]
    unexpected_missing = [
        k for k in incompatible.missing_keys if not k.startswith(allowed)
    ]

    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' does not match this model.\n"
            f"  Missing from checkpoint ({len(unexpected_missing)}): "
            f"{unexpected_missing[:10]}\n"
            f"  Not present in model ({len(incompatible.unexpected_keys)}): "
            f"{incompatible.unexpected_keys[:10]}\n"
            f"Loading anyway would leave those layers at random init. Use a "
            f"checkpoint from this architecture, or pass allow_missing=(...) "
            f"if the caller handles the absent weights."
        )

    if expected_missing:
        logger.warning(
            f"Checkpoint '{checkpoint_path}' predates {len(expected_missing)} "
            f"parameter(s) this model now has: {expected_missing[:5]}. "
            f"Left uninitialized at the caller's request."
        )

    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            logger.warning("Could not restore optimizer state (architecture changed).")

    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('loss', None)
    loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
    logger.info(
        f"Checkpoint loaded: {checkpoint_path} (epoch {epoch}, loss {loss_str})"
    )

    return checkpoint
