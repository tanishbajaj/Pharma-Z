"""
Binding Affinity Service

Business logic for drug-protein binding affinity prediction using DeepDTA.
Handles data loading, training, and prediction.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.config import settings
from app.models.deep_dta_model import (
    DeepDTA,
    encode_drug_sequence,
    encode_protein_sequence,
    DRUG_MAX_LENGTH,
    PROTEIN_MAX_LENGTH,
)
from app.models.training_utils import (
    get_training_device,
    get_mixed_precision_context,
    get_gradient_scaler,
    save_model_checkpoint,
    load_model_checkpoint,
)
from app.utils.logging_config import logger
from app.utils.constants import KIBA_FILENAME


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

_dta_model: Optional[DeepDTA] = None
_device: Optional[torch.device] = None


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_kiba_dataset(
    file_path: str = None,
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Load the KIBA binding affinity dataset.

    Expected columns: smiles, protein_sequence, affinity_score

    Args:
        file_path:   Path to KIBA CSV file.
        max_samples: Limit number of samples loaded.

    Returns:
        DataFrame with drug-protein-affinity data.
    """
    if file_path is None:
        file_path = os.path.join(settings.data_raw_dir, KIBA_FILENAME)

    logger.info(f"Loading KIBA dataset from: {file_path}")
    dataframe = pd.read_csv(file_path)

    if max_samples is not None:
        dataframe = dataframe.head(max_samples)

    logger.info(f"Loaded {len(dataframe)} drug-target pairs")
    return dataframe


def prepare_binding_data(
    drug_smiles_list: List[str],
    protein_sequence_list: List[str],
    affinity_scores: List[float] = None,
    batch_size: int = None,
) -> DataLoader:
    """
    Encode drug-protein pairs and create a DataLoader.

    Args:
        drug_smiles_list:     List of drug SMILES strings.
        protein_sequence_list: List of protein amino acid sequences.
        affinity_scores:      Target affinity values (None for prediction-only).
        batch_size:           Batch size.

    Returns:
        DataLoader yielding (drug_tokens, protein_tokens[, affinity]) batches.
    """
    if batch_size is None:
        batch_size = settings.default_batch_size

    # Encode sequences
    drug_encoded = [encode_drug_sequence(s) for s in drug_smiles_list]
    protein_encoded = [encode_protein_sequence(p) for p in protein_sequence_list]

    drug_tensor = torch.tensor(drug_encoded, dtype=torch.long)
    protein_tensor = torch.tensor(protein_encoded, dtype=torch.long)

    if affinity_scores is not None:
        affinity_tensor = torch.tensor(affinity_scores, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(drug_tensor, protein_tensor, affinity_tensor)
    else:
        dataset = TensorDataset(drug_tensor, protein_tensor)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(affinity_scores is not None),
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Prepared binding DataLoader: {len(dataset)} pairs, batch_size={batch_size}")
    return data_loader


# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------

def initialize_dta_model() -> DeepDTA:
    """Create and return a new DeepDTA model on the best device."""
    global _dta_model, _device

    _device = get_training_device()
    _dta_model = DeepDTA().to(_device)

    total_params = sum(p.numel() for p in _dta_model.parameters())
    logger.info(f"DeepDTA initialized: {total_params:,} parameters")

    return _dta_model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deep_dta(
    data_loader: DataLoader,
    model: DeepDTA = None,
    num_epochs: int = None,
    learning_rate: float = None,
    checkpoint_dir: str = None,
) -> dict:
    """
    Train the DeepDTA model.

    Args:
        data_loader:    DataLoader with (drug, protein, affinity) batches.
        model:          DeepDTA model (creates new if None).
        num_epochs:     Number of epochs.
        learning_rate:  Optimizer learning rate.
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        Training history dict with loss per epoch.
    """
    if model is None:
        model = initialize_dta_model()

    if num_epochs is None:
        num_epochs = settings.default_epochs

    if learning_rate is None:
        learning_rate = settings.default_learning_rate

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(settings.saved_models_dir, "deep_dta")

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    scaler = get_gradient_scaler(device)

    training_history = {"mse_loss": []}

    logger.info(f"Starting DeepDTA training: epochs={num_epochs}, lr={learning_rate}")
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            drug_tokens = batch[0].to(device)
            protein_tokens = batch[1].to(device)
            target_affinity = batch[2].to(device)

            optimizer.zero_grad()

            with get_mixed_precision_context(device):
                predicted_affinity = model(drug_tokens, protein_tokens)
                loss = loss_function(predicted_affinity, target_affinity)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        training_history["mse_loss"].append(avg_loss)

        if epoch % 10 == 0 or epoch == num_epochs:
            logger.info(f"Epoch {epoch}/{num_epochs} | MSE: {avg_loss:.4f}")
            checkpoint_path = os.path.join(checkpoint_dir, f"dta_epoch_{epoch}.pt")
            save_model_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

    logger.info("DeepDTA training complete.")
    return training_history


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_binding_affinity(
    drug_smiles: str,
    protein_sequence: str,
    model: DeepDTA = None,
    checkpoint_path: str = None,
) -> dict:
    """
    Predict binding affinity for a single drug-protein pair.

    Args:
        drug_smiles:     Drug SMILES string.
        protein_sequence: Protein amino acid sequence.
        model:           Trained DeepDTA model.
        checkpoint_path: Path to model checkpoint.

    Returns:
        Dict with drug_smiles, protein_sequence, binding_affinity_score.
    """
    global _dta_model, _device

    if model is None:
        if _dta_model is None:
            if checkpoint_path:
                _dta_model = initialize_dta_model()
                load_model_checkpoint(_dta_model, checkpoint_path)
            else:
                raise ValueError("No model available. Train first or provide checkpoint.")
        model = _dta_model

    device = next(model.parameters()).device
    model.eval()

    drug_encoded = torch.tensor(
        [encode_drug_sequence(drug_smiles)], dtype=torch.long, device=device
    )
    protein_encoded = torch.tensor(
        [encode_protein_sequence(protein_sequence)], dtype=torch.long, device=device
    )

    with torch.no_grad():
        predicted_score = model(drug_encoded, protein_encoded)

    return {
        "drug_smiles": drug_smiles,
        "protein_sequence": protein_sequence[:50] + "..." if len(protein_sequence) > 50 else protein_sequence,
        "binding_affinity_score": round(float(predicted_score.item()), 4),
    }


def predict_binding_batch(
    drug_smiles_list: List[str],
    protein_sequence_list: List[str],
    model: DeepDTA = None,
) -> List[dict]:
    """
    Predict binding affinity for multiple drug-protein pairs.

    Args:
        drug_smiles_list:      List of drug SMILES.
        protein_sequence_list: List of protein sequences.
        model:                 Trained DeepDTA model.

    Returns:
        List of prediction dicts.
    """
    results = []
    for drug, protein in zip(drug_smiles_list, protein_sequence_list):
        prediction = predict_binding_affinity(drug, protein, model=model)
        results.append(prediction)

    return results
