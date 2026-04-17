"""
Binding Affinity Schemas

Pydantic models for binding affinity prediction endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class BindingPredictRequest(BaseModel):
    """Request to predict binding affinity for a drug-protein pair."""
    drug_smiles: str = Field(
        ...,
        description="SMILES string of the drug molecule",
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
    )
    protein_sequence: str = Field(
        ...,
        description="Amino acid sequence of the target protein",
        examples=["MKTAYIAKQRQISFVKSHFSRQLE"],
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to trained DeepDTA checkpoint",
    )


class BindingBatchPredictRequest(BaseModel):
    """Request for batch binding affinity prediction."""
    drug_smiles_list: List[str] = Field(..., description="List of drug SMILES")
    protein_sequence_list: List[str] = Field(..., description="List of protein sequences")


class BindingTrainRequest(BaseModel):
    """Request to train the DeepDTA model."""
    dataset_name: str = Field(default="kiba", description="Dataset: 'kiba' or 'davis'")
    dataset_path: Optional[str] = Field(default=None, description="Override dataset path")
    model_type: str = Field(
        default="baseline",
        description="Model type: 'baseline' (CNN) or 'esm' (ESM-enhanced)",
    )
    max_samples: Optional[int] = Field(default=None, description="Limit training samples")
    num_epochs: int = Field(default=100, ge=1, le=500, description="Training epochs (paper: 100)")
    batch_size: int = Field(default=256, ge=8, le=512, description="Batch size (paper: 256)")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate (paper: 0.001)")
    split_mode: str = Field(
        default="random",
        description="Split strategy: 'random', 'cold-drug', 'cold-protein', 'fold'",
    )
    patience: int = Field(default=20, ge=0, description="Early stopping patience (0=disabled)")


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class BindingPredictResponse(BaseModel):
    """Response for binding affinity prediction."""
    drug_smiles: str
    protein_sequence: str
    binding_affinity_score: float


class BindingBatchPredictResponse(BaseModel):
    """Response for batch binding prediction."""
    predictions: List[BindingPredictResponse]
    total_predictions: int


class BindingTrainMetrics(BaseModel):
    """Evaluation metrics from training."""
    mse: float = Field(description="Mean Squared Error")
    rmse: float = Field(description="Root Mean Squared Error")
    concordance_index: float = Field(description="Concordance Index (CI) — primary metric")
    pearson_r: float = Field(description="Pearson correlation coefficient")


class BindingTrainResponse(BaseModel):
    """Response for binding affinity training."""
    status: str = Field(default="success")
    model_type: str = Field(description="Model variant used: 'baseline' or 'esm'")
    dataset: str = Field(description="Dataset trained on")
    epochs_completed: int = Field(description="Number of epochs completed")
    best_val_ci: float = Field(description="Best validation Concordance Index achieved")
    test_metrics: Optional[BindingTrainMetrics] = Field(
        default=None, description="Final test set metrics"
    )
    training_time_seconds: float = Field(description="Total training time in seconds")
    checkpoint_path: str = Field(description="Path to best model checkpoint")
