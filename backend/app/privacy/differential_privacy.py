"""
Differential Privacy Module

Wrapper around Opacus for applying differential privacy guarantees
to PyTorch model training.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.utils.logging_config import logger

# Opacus is an optional dependency
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logger.warning("Opacus not installed — differential privacy unavailable")


def make_model_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    target_epsilon: float = None,
    target_delta: float = 1e-5,
    epochs: int = 50,
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
    """
    Wrap a PyTorch model with Opacus PrivacyEngine for DP-SGD training.

    Args:
        model:            PyTorch model to make private.
        optimizer:        Optimizer for the model.
        data_loader:      Training DataLoader.
        noise_multiplier: Noise added to gradients (higher = more private).
        max_grad_norm:    Maximum gradient norm for clipping.
        target_epsilon:   Target privacy budget (if set, auto-calibrates noise).
        target_delta:     Target delta parameter.
        epochs:           Expected training epochs (for epsilon computation).

    Returns:
        Tuple of (private_model, private_optimizer, private_data_loader).
    """
    if not OPACUS_AVAILABLE:
        logger.warning("Opacus not available — returning model without DP")
        return model, optimizer, data_loader

    privacy_engine = PrivacyEngine()

    if target_epsilon is not None:
        # Auto-calibrate noise to achieve target epsilon
        private_model, private_optimizer, private_data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )
        logger.info(
            f"DP enabled (auto-calibrated): target_ε={target_epsilon}, "
            f"δ={target_delta}, noise≈{private_optimizer.noise_multiplier:.2f}"
        )
    else:
        private_model, private_optimizer, private_data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        logger.info(
            f"DP enabled: noise_multiplier={noise_multiplier}, "
            f"max_grad_norm={max_grad_norm}"
        )

    return private_model, private_optimizer, private_data_loader


def get_privacy_spent(
    privacy_engine,
    delta: float = 1e-5,
) -> dict:
    """
    Get the current privacy budget spent during training.

    Returns:
        Dict with epsilon and delta values.
    """
    if not OPACUS_AVAILABLE:
        return {"epsilon": float("inf"), "delta": delta, "available": False}

    try:
        epsilon = privacy_engine.get_epsilon(delta=delta)
        return {
            "epsilon": round(epsilon, 4),
            "delta": delta,
            "available": True,
        }
    except Exception as error:
        logger.error(f"Failed to compute privacy budget: {error}")
        return {"epsilon": float("inf"), "delta": delta, "available": False}
