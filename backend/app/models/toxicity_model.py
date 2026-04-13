"""
Toxicity Prediction Model

Random Forest classifier using Morgan molecular fingerprints (RDKit).
Trained on Tox21 dataset for multi-label toxicity classification.
"""

import numpy as np
from typing import List, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 2048
N_ESTIMATORS = 100
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """
    Convert a SMILES string to a Morgan fingerprint vector.

    Args:
        smiles: SMILES molecular string.

    Returns:
        Numpy array of shape (FINGERPRINT_NBITS,) or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_NBITS
    )

    return np.array(fingerprint, dtype=np.float32)


def batch_smiles_to_fingerprints(smiles_list: List[str]) -> tuple:
    """
    Convert a batch of SMILES strings to fingerprint feature matrix.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Tuple of (feature_matrix, valid_indices, invalid_indices).
        feature_matrix: np.ndarray of shape (num_valid, FINGERPRINT_NBITS)
    """
    features = []
    valid_indices = []
    invalid_indices = []

    for index, smiles in enumerate(smiles_list):
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is not None:
            features.append(fingerprint)
            valid_indices.append(index)
        else:
            invalid_indices.append(index)

    if features:
        feature_matrix = np.stack(features)
    else:
        feature_matrix = np.empty((0, FINGERPRINT_NBITS), dtype=np.float32)

    return feature_matrix, valid_indices, invalid_indices


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ToxicityPredictor:
    """
    Random Forest-based toxicity prediction model.

    Uses Morgan fingerprints as features. Supports both single-label
    and multi-label (Tox21-style) classification.
    """

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        random_state: int = RANDOM_STATE,
        multi_label: bool = True,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.multi_label = multi_label
        self.model = None
        self.label_names: List[str] = []
        self.is_trained = False

    def build_model(self, num_labels: int = 1):
        """Initialize the Random Forest classifier."""
        base_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        if self.multi_label and num_labels > 1:
            self.model = MultiOutputClassifier(base_classifier)
        else:
            self.model = base_classifier

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        label_names: List[str] = None,
    ):
        """
        Train the toxicity model.

        Args:
            features:    (num_samples, FINGERPRINT_NBITS) fingerprint matrix.
            labels:      (num_samples,) or (num_samples, num_labels) toxicity labels.
            label_names: Names of the toxicity assays (e.g., Tox21 columns).
        """
        if labels.ndim == 1:
            num_labels = 1
        else:
            num_labels = labels.shape[1]

        if label_names:
            self.label_names = label_names
        else:
            self.label_names = [f"toxicity_{i}" for i in range(num_labels)]

        self.build_model(num_labels)
        self.model.fit(features, labels)
        self.is_trained = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict toxicity labels.

        Args:
            features: (num_samples, FINGERPRINT_NBITS) fingerprint matrix.

        Returns:
            Predicted labels: (num_samples,) or (num_samples, num_labels).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict toxicity probabilities.

        Args:
            features: (num_samples, FINGERPRINT_NBITS) fingerprint matrix.

        Returns:
            Probability array. For multi-label, returns probability of
            positive class for each label: (num_samples, num_labels).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        if self.multi_label and hasattr(self.model, "estimators_"):
            # MultiOutputClassifier returns list of arrays
            probas = self.model.predict_proba(features)
            # Extract probability of positive class (index 1) for each label
            result = np.column_stack([
                p[:, 1] if p.shape[1] > 1 else p[:, 0]
                for p in probas
            ])
            return result
        else:
            probas = self.model.predict_proba(features)
            if probas.shape[1] == 2:
                return probas[:, 1]  # Probability of toxic (positive class)
            return probas

    def predict_single_smiles(self, smiles: str) -> dict:
        """
        Predict toxicity for a single SMILES string.

        Returns:
            Dict with is_valid, toxicity_probability, label_probabilities.
        """
        result = {
            "smiles": smiles,
            "is_valid": False,
            "is_toxic": False,
            "toxicity_probability": 0.0,
            "label_probabilities": {},
        }

        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is None:
            return result

        result["is_valid"] = True

        features = fingerprint.reshape(1, -1)
        probabilities = self.predict_proba(features)

        if probabilities.ndim == 1:
            # Single label
            tox_prob = float(probabilities[0])
            result["toxicity_probability"] = round(tox_prob, 4)
            result["is_toxic"] = tox_prob > 0.5
        else:
            # Multi-label
            avg_prob = float(np.mean(probabilities[0]))
            result["toxicity_probability"] = round(avg_prob, 4)
            result["is_toxic"] = avg_prob > 0.5
            result["label_probabilities"] = {
                name: round(float(prob), 4)
                for name, prob in zip(self.label_names, probabilities[0])
            }

        return result
