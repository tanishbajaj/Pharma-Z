"""
Toxicity Service

Business logic for toxicity prediction using Random Forest on Morgan fingerprints.
Handles Tox21 data loading, model training, and prediction.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

from app.config import settings
from app.models.toxicity_model import (
    ToxicityPredictor,
    smiles_to_fingerprint,
    batch_smiles_to_fingerprints,
    FINGERPRINT_NBITS,
)
from app.utils.logging_config import logger
from app.utils.constants import TOX21_FILENAME


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

_toxicity_model: Optional[ToxicityPredictor] = None

# Tox21 assay column names
TOX21_LABEL_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_tox21_dataset(
    file_path: str = None,
    max_samples: int = None,
) -> tuple:
    """
    Load and prepare the Tox21 dataset.

    Returns:
        Tuple of (smiles_list, labels_matrix, label_names).
        labels_matrix: np.ndarray of shape (num_samples, 12) with 0/1 values.
    """
    if file_path is None:
        file_path = os.path.join(settings.data_raw_dir, TOX21_FILENAME)

    logger.info(f"Loading Tox21 dataset from: {file_path}")
    dataframe = pd.read_csv(file_path)

    # Find SMILES column
    smiles_column = None
    for candidate in ["smiles", "SMILES", "Smiles", "mol"]:
        if candidate in dataframe.columns:
            smiles_column = candidate
            break
    if smiles_column is None:
        smiles_column = dataframe.columns[0]

    # Find available label columns
    available_labels = [col for col in TOX21_LABEL_COLUMNS if col in dataframe.columns]
    if not available_labels:
        # Fallback: use numeric columns (excluding the SMILES column)
        available_labels = [
            col for col in dataframe.columns
            if col != smiles_column and dataframe[col].dtype in ["float64", "int64"]
        ]

    # Drop rows with missing SMILES
    dataframe = dataframe.dropna(subset=[smiles_column])

    if max_samples is not None:
        dataframe = dataframe.head(max_samples)

    smiles_list = dataframe[smiles_column].tolist()

    # Fill NaN labels with 0 (inactive)
    labels_matrix = dataframe[available_labels].fillna(0).values.astype(np.float32)

    logger.info(
        f"Loaded Tox21: {len(smiles_list)} molecules, "
        f"{len(available_labels)} assays: {available_labels}"
    )

    return smiles_list, labels_matrix, available_labels


def prepare_toxicity_data(
    smiles_list: List[str],
    labels: np.ndarray = None,
) -> tuple:
    """
    Convert SMILES to fingerprint features and filter invalid molecules.

    Returns:
        Tuple of (features, filtered_labels, valid_count, invalid_count).
    """
    features, valid_indices, invalid_indices = batch_smiles_to_fingerprints(smiles_list)

    filtered_labels = None
    if labels is not None:
        filtered_labels = labels[valid_indices]

    logger.info(
        f"Feature extraction: {len(valid_indices)} valid, "
        f"{len(invalid_indices)} invalid molecules"
    )

    return features, filtered_labels, len(valid_indices), len(invalid_indices)


# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------

def initialize_toxicity_model(multi_label: bool = True) -> ToxicityPredictor:
    """Create a new toxicity predictor instance."""
    global _toxicity_model
    _toxicity_model = ToxicityPredictor(multi_label=multi_label)
    logger.info("Toxicity predictor initialized")
    return _toxicity_model


def get_toxicity_model() -> ToxicityPredictor:
    """Get or create the toxicity model."""
    global _toxicity_model
    if _toxicity_model is None:
        _toxicity_model = ToxicityPredictor()
    return _toxicity_model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_toxicity_model(
    smiles_list: List[str] = None,
    labels: np.ndarray = None,
    label_names: List[str] = None,
    dataset_path: str = None,
    max_samples: int = None,
    save_path: str = None,
) -> dict:
    """
    Train the toxicity prediction model.

    Can either use provided data or load from Tox21 dataset.

    Args:
        smiles_list:  List of SMILES strings.
        labels:       Toxicity labels (num_samples, num_labels).
        label_names:  Label column names.
        dataset_path: Path to Tox21 CSV (used if smiles_list is None).
        max_samples:  Limit training samples.
        save_path:    Path to save trained model.

    Returns:
        Dict with training results.
    """
    # Load data if not provided
    if smiles_list is None:
        smiles_list, labels, label_names = load_tox21_dataset(
            file_path=dataset_path, max_samples=max_samples
        )

    # Convert to features
    features, filtered_labels, valid_count, invalid_count = prepare_toxicity_data(
        smiles_list, labels
    )

    if valid_count == 0:
        raise ValueError("No valid molecules found in the dataset.")

    # Train
    model = initialize_toxicity_model(multi_label=(filtered_labels.ndim > 1))
    model.train(features, filtered_labels, label_names=label_names)

    # Save model
    if save_path is None:
        save_dir = os.path.join(settings.saved_models_dir, "toxicity")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "toxicity_rf_model.pkl")

    joblib.dump(model, save_path)
    logger.info(f"Toxicity model saved to: {save_path}")

    result = {
        "status": "completed",
        "total_samples": len(smiles_list),
        "valid_samples": valid_count,
        "invalid_samples": invalid_count,
        "num_labels": len(label_names) if label_names else 1,
        "label_names": label_names or [],
        "model_path": save_path,
    }

    return result


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_toxicity(
    smiles: str,
    model: ToxicityPredictor = None,
    model_path: str = None,
) -> dict:
    """
    Predict toxicity for a single SMILES string.

    Args:
        smiles:     SMILES string.
        model:      Trained ToxicityPredictor.
        model_path: Path to saved model file.

    Returns:
        Dict with toxicity prediction results.
    """
    global _toxicity_model

    if model is None:
        if _toxicity_model is None or not _toxicity_model.is_trained:
            if model_path:
                _toxicity_model = joblib.load(model_path)
                logger.info(f"Loaded toxicity model from: {model_path}")
            else:
                raise ValueError("No model available. Train first or provide model_path.")
        model = _toxicity_model

    return model.predict_single_smiles(smiles)


def predict_toxicity_batch(
    smiles_list: List[str],
    model: ToxicityPredictor = None,
) -> List[dict]:
    """Predict toxicity for a batch of SMILES strings."""
    return [predict_toxicity(smiles, model=model) for smiles in smiles_list]
