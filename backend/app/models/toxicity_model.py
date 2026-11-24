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

class MaskedMultiOutputRF:
    """
    One Random Forest per assay, each trained ONLY on measured labels.

    Tox21 tests each molecule on a subset of the 12 assays, leaving ~17% of
    label cells missing. scikit-learn's MultiOutputClassifier cannot mask
    per-output, so the usual workaround is to fill missing entries with 0 —
    which silently invents ~16,000 "non-toxic" labels. Ebner et al. (2025),
    arXiv:2511.14744, identify that zero-filling as a source of benchmark
    drift on Tox21 and recommend masking instead.

    This class exposes `estimators_` and a list-returning `predict_proba`,
    so it is drop-in compatible with how ToxicityPredictor consumes a
    MultiOutputClassifier.
    """

    def __init__(self, n_estimators: int = N_ESTIMATORS, random_state: int = RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_: List[RandomForestClassifier] = []
        self.n_train_per_assay: List[int] = []

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (n, n_bits) fingerprints.
            labels:   (n, k) with np.nan where the assay was not measured.
        """
        self.estimators_ = []
        self.n_train_per_assay = []

        for index in range(labels.shape[1]):
            measured = ~np.isnan(labels[:, index])
            classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            classifier.fit(features[measured], labels[measured, index].astype(int))
            self.estimators_.append(classifier)
            self.n_train_per_assay.append(int(measured.sum()))

        return self

    def predict_proba(self, features: np.ndarray) -> list:
        """Mirrors MultiOutputClassifier: a list of (n, 2) arrays."""
        return [estimator.predict_proba(features) for estimator in self.estimators_]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.column_stack([e.predict(features) for e in self.estimators_])


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
        aggregation: str = "max",
        toxic_threshold: float = 0.5,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.multi_label = multi_label
        # How 12 per-assay probabilities collapse into one screening score,
        # and the cutoff for is_toxic. Both are chosen on a validation split
        # by scripts/train_toxicity.py — "mean > 0.5" never fires, because
        # per-assay positive rates are only 3-16%.
        self.aggregation = aggregation
        self.toxic_threshold = toxic_threshold
        self.model = None
        self.label_names: List[str] = []
        self.is_trained = False

    def train_masked(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        label_names: List[str] = None,
    ):
        """
        Train one RF per assay on measured labels only (NaN = not measured).
        """
        self.label_names = label_names or [
            f"toxicity_{i}" for i in range(labels.shape[1])
        ]
        self.multi_label = True
        self.model = MaskedMultiOutputRF(
            n_estimators=self.n_estimators, random_state=self.random_state
        ).fit(features, labels)
        self.is_trained = True
        return self

    def aggregate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Collapse per-assay probabilities (n, k) into one score per molecule.

        max      — a molecule strongly toxic on ONE assay is toxic; averaging
                   that against 11 clean assays dilutes the signal away.
        noisy_or — 1 - prod(1 - p), treats assays as independent evidence.
        mean     — the original behaviour, kept for comparison.
        """
        mode = getattr(self, "aggregation", "mean")
        if mode == "max":
            return probabilities.max(axis=1)
        if mode == "noisy_or":
            return 1.0 - np.prod(1.0 - probabilities, axis=1)
        return probabilities.mean(axis=1)

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
            # Multi-label: aggregate with the configured rule and threshold.
            score = float(self.aggregate(probabilities)[0])
            threshold = getattr(self, "toxic_threshold", 0.5)
            result["toxicity_probability"] = round(score, 4)
            result["is_toxic"] = score > threshold
            result["aggregation"] = getattr(self, "aggregation", "mean")
            result["threshold"] = threshold
            result["label_probabilities"] = {
                name: round(float(prob), 4)
                for name, prob in zip(self.label_names, probabilities[0])
            }

        return result
