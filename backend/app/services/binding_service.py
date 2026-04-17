"""
Binding Affinity Service

Business logic for drug-protein binding affinity prediction using DeepDTA.
Supports two model tiers:
  - Baseline: DeepDTA CNN (paper-accurate)
  - ESM-Enhanced: DeepDTA with ESM-2 protein embeddings

Handles data loading, train/val/test splitting, training with validation,
evaluation, and prediction.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

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
from app.models.binding_metrics import compute_all_metrics, concordance_index_fast
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

_dta_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_binding_dataset(
    dataset_name: str = "kiba",
    file_path: str = None,
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Load a binding affinity dataset (KIBA or Davis).

    Expected columns: smiles, protein_sequence, affinity_score

    Args:
        dataset_name: 'kiba' or 'davis'
        file_path:    Override path to CSV file.
        max_samples:  Limit number of samples loaded.

    Returns:
        DataFrame with drug-protein-affinity data.
    """
    if file_path is None:
        filename = f"{dataset_name}.csv"
        file_path = os.path.join(settings.data_raw_dir, filename)

    logger.info(f"Loading {dataset_name.upper()} dataset from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            f"Run: python scripts/download_binding_data.py --dataset {dataset_name}"
        )

    dataframe = pd.read_csv(file_path)

    if max_samples is not None:
        dataframe = dataframe.head(max_samples)

    logger.info(f"Loaded {len(dataframe)} drug-target pairs")
    return dataframe


# Legacy alias for backward compatibility
def load_kiba_dataset(file_path=None, max_samples=None):
    return load_binding_dataset("kiba", file_path, max_samples)


# ---------------------------------------------------------------------------
# Data Splitting
# ---------------------------------------------------------------------------

def create_train_val_test_split(
    df: pd.DataFrame,
    split_mode: str = "random",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    fold_num: int = 0,
    folds_path: str = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test sets.

    Args:
        df:          Full dataset DataFrame.
        split_mode:  'random', 'cold-drug', 'cold-protein', or 'fold'
        val_ratio:   Fraction for validation (ignored if split_mode='fold')
        test_ratio:  Fraction for test (ignored if split_mode='fold')
        fold_num:    Which fold to use as validation (0-4) for 'fold' mode
        folds_path:  Path to folds JSON file (for 'fold' mode)
        seed:        Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    rng = np.random.RandomState(seed)

    if split_mode == "random":
        indices = rng.permutation(n)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)

        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]

    elif split_mode == "cold-drug":
        # Test set contains drugs not seen in training
        unique_drugs = df["smiles"].unique()
        rng.shuffle(unique_drugs)
        test_drugs = set(unique_drugs[:int(len(unique_drugs) * test_ratio)])
        val_drugs = set(unique_drugs[int(len(unique_drugs) * test_ratio):int(len(unique_drugs) * (test_ratio + val_ratio))])

        test_indices = df[df["smiles"].isin(test_drugs)].index.values
        val_indices = df[df["smiles"].isin(val_drugs)].index.values
        train_indices = df[~df["smiles"].isin(test_drugs | val_drugs)].index.values

    elif split_mode == "cold-protein":
        # Test set contains proteins not seen in training
        unique_proteins = df["protein_sequence"].unique()
        rng.shuffle(unique_proteins)
        test_proteins = set(unique_proteins[:int(len(unique_proteins) * test_ratio)])
        val_proteins = set(unique_proteins[int(len(unique_proteins) * test_ratio):int(len(unique_proteins) * (test_ratio + val_ratio))])

        test_indices = df[df["protein_sequence"].isin(test_proteins)].index.values
        val_indices = df[df["protein_sequence"].isin(val_proteins)].index.values
        train_indices = df[~df["protein_sequence"].isin(test_proteins | val_proteins)].index.values

    elif split_mode == "fold":
        # Use DeepDTA's official 5-fold indices
        if folds_path is None:
            raise ValueError("folds_path required for fold split mode")

        with open(folds_path) as f:
            folds_data = json.load(f)

        train_folds = folds_data["train_folds"]
        test_fold = folds_data["test_fold"]

        test_indices = np.array(test_fold)
        # Use one of the 5 train folds as validation
        val_indices = np.array(train_folds[fold_num])
        train_indices = np.concatenate([
            np.array(train_folds[i]) for i in range(len(train_folds)) if i != fold_num
        ])

    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    logger.info(f"Split ({split_mode}): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# DataLoader Preparation
# ---------------------------------------------------------------------------

def prepare_binding_data(
    drug_smiles_list: List[str],
    protein_sequence_list: List[str],
    affinity_scores: List[float] = None,
    batch_size: int = None,
    shuffle: bool = True,
    protein_embeddings: Dict[str, torch.Tensor] = None,
) -> DataLoader:
    """
    Encode drug-protein pairs and create a DataLoader.

    Args:
        drug_smiles_list:      List of drug SMILES strings.
        protein_sequence_list: List of protein amino acid sequences.
        affinity_scores:       Target affinity values (None for prediction-only).
        batch_size:            Batch size.
        shuffle:               Whether to shuffle data.
        protein_embeddings:    Pre-computed ESM embeddings {sequence: tensor}.

    Returns:
        DataLoader yielding batches.
    """
    if batch_size is None:
        batch_size = settings.default_batch_size

    # Encode drug sequences
    drug_encoded = [encode_drug_sequence(s) for s in drug_smiles_list]
    drug_tensor = torch.tensor(drug_encoded, dtype=torch.long)

    if protein_embeddings is not None:
        # ESM mode: use pre-computed embeddings
        emb_list = []
        for seq in protein_sequence_list:
            if seq in protein_embeddings:
                emb_list.append(protein_embeddings[seq])
            else:
                # Fallback: zero embedding (should not happen if cache is complete)
                emb_dim = next(iter(protein_embeddings.values())).shape[0]
                emb_list.append(torch.zeros(emb_dim))
        protein_tensor = torch.stack(emb_list)

        if affinity_scores is not None:
            affinity_tensor = torch.tensor(affinity_scores, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(drug_tensor, protein_tensor, affinity_tensor)
        else:
            dataset = TensorDataset(drug_tensor, protein_tensor)
    else:
        # Baseline mode: encode protein sequences as integer tokens
        protein_encoded = [encode_protein_sequence(p) for p in protein_sequence_list]
        protein_tensor = torch.tensor(protein_encoded, dtype=torch.long)

        if affinity_scores is not None:
            affinity_tensor = torch.tensor(affinity_scores, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(drug_tensor, protein_tensor, affinity_tensor)
        else:
            dataset = TensorDataset(drug_tensor, protein_tensor)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
    )

    logger.info(f"Prepared binding DataLoader: {len(dataset)} pairs, batch_size={batch_size}")
    return data_loader


# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------

def initialize_dta_model(model_type: str = "baseline", **kwargs) -> torch.nn.Module:
    """
    Create and return a new DeepDTA model on the best device.

    Args:
        model_type: 'baseline' or 'esm'
        **kwargs:   Extra model parameters.

    Returns:
        Initialized model on device.
    """
    global _dta_model, _device

    _device = get_training_device()

    if model_type == "esm":
        from app.models.deep_dta_esm import DeepDTA_ESM
        _dta_model = DeepDTA_ESM(**kwargs).to(_device)
    else:
        _dta_model = DeepDTA(**kwargs).to(_device)

    total_params = sum(p.numel() for p in _dta_model.parameters())
    logger.info(f"DeepDTA ({model_type}) initialized: {total_params:,} parameters on {_device}")

    return _dta_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on a dataset.

    Args:
        model:       Trained model.
        data_loader: DataLoader with (drug, protein, affinity) batches.
        device:      Device to run on.

    Returns:
        Dict with keys: mse, rmse, ci, pearson_r
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            drug_tokens = batch[0].to(device)
            protein_data = batch[1].to(device)
            target_affinity = batch[2].to(device)

            predicted = model(drug_tokens, protein_data)

            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(target_affinity.cpu().numpy())

    all_predictions = np.concatenate(all_predictions).ravel()
    all_targets = np.concatenate(all_targets).ravel()

    metrics = compute_all_metrics(all_targets, all_predictions)

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deep_dta(
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    model: torch.nn.Module = None,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    checkpoint_dir: str = None,
    patience: int = 20,
) -> dict:
    """
    Train the DeepDTA model with validation tracking and early stopping.

    Args:
        train_loader:   DataLoader with training data.
        val_loader:     DataLoader with validation data (optional).
        model:          DeepDTA model (creates new if None).
        num_epochs:     Number of epochs.
        learning_rate:  Optimizer learning rate.
        checkpoint_dir: Directory to save checkpoints.
        patience:       Early stopping patience (epochs without CI improvement).

    Returns:
        Training history dict with metrics per epoch.
    """
    if model is None:
        model = initialize_dta_model()

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(settings.saved_models_dir, "deep_dta")

    os.makedirs(checkpoint_dir, exist_ok=True)

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    loss_function = torch.nn.MSELoss()
    scaler = get_gradient_scaler(device)

    training_history = {
        "train_mse": [],
        "val_mse": [],
        "val_rmse": [],
        "val_ci": [],
        "val_pearson_r": [],
        "lr": [],
    }

    best_ci = 0.0
    epochs_without_improvement = 0

    logger.info(f"Starting DeepDTA training: epochs={num_epochs}, lr={learning_rate}")
    model.train()

    for epoch in range(1, num_epochs + 1):
        # ------- Training -------
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            drug_tokens = batch[0].to(device)
            protein_data = batch[1].to(device)
            target_affinity = batch[2].to(device)

            optimizer.zero_grad()

            with get_mixed_precision_context(device):
                predicted_affinity = model(drug_tokens, protein_data)
                loss = loss_function(predicted_affinity, target_affinity)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        training_history["train_mse"].append(avg_train_loss)
        training_history["lr"].append(optimizer.param_groups[0]["lr"])

        # ------- Validation -------
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device)
            model.train()  # Switch back to training mode

            training_history["val_mse"].append(val_metrics["mse"])
            training_history["val_rmse"].append(val_metrics["rmse"])
            training_history["val_ci"].append(val_metrics["ci"])
            training_history["val_pearson_r"].append(val_metrics["pearson_r"])

            # Update LR scheduler
            scheduler.step(val_metrics["mse"])

            # Check for improvement
            if val_metrics["ci"] > best_ci:
                best_ci = val_metrics["ci"]
                epochs_without_improvement = 0

                # Save best model
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_model_checkpoint(model, optimizer, epoch, avg_train_loss, best_path)
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train MSE: {avg_train_loss:.4f} | "
                    f"Val MSE: {val_metrics['mse']:.4f} | "
                    f"Val CI: {val_metrics['ci']:.4f} ★ (best)"
                )
            else:
                epochs_without_improvement += 1
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{num_epochs} | "
                        f"Train MSE: {avg_train_loss:.4f} | "
                        f"Val MSE: {val_metrics['mse']:.4f} | "
                        f"Val CI: {val_metrics['ci']:.4f}"
                    )

            # Early stopping
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no CI improvement for {patience} epochs)")
                break
        else:
            if epoch % 10 == 0 or epoch == num_epochs:
                logger.info(f"Epoch {epoch}/{num_epochs} | Train MSE: {avg_train_loss:.4f}")

        # Periodic checkpoint
        if epoch % 25 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dta_epoch_{epoch}.pt")
            save_model_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_path)

    # Load best model if we have validation
    if val_loader is not None:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            load_model_checkpoint(model, best_path)
            logger.info(f"Loaded best model (CI={best_ci:.4f})")

    logger.info("DeepDTA training complete.")
    return training_history


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_binding_affinity(
    drug_smiles: str,
    protein_sequence: str,
    model: torch.nn.Module = None,
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
    model: torch.nn.Module = None,
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
