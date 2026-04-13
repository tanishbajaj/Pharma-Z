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
    dataset_path: Optional[str] = Field(default=None)
    max_samples: Optional[int] = Field(default=None)
    num_epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=64, ge=8, le=512)
    learning_rate: float = Field(default=0.001, gt=0)


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
