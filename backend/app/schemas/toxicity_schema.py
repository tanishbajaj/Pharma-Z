"""
Toxicity Schemas

Pydantic models for toxicity prediction endpoints.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class ToxicityPredictRequest(BaseModel):
    """Request to predict toxicity of a molecule."""
    smiles: str = Field(
        ..., description="SMILES string of the molecule",
        examples=["CCO"],
    )
    model_path: Optional[str] = Field(default=None, description="Path to saved toxicity model")


class ToxicityBatchPredictRequest(BaseModel):
    """Request for batch toxicity prediction."""
    smiles_list: List[str] = Field(..., description="List of SMILES strings")


class ToxicityTrainRequest(BaseModel):
    """Request to train the toxicity model."""
    dataset_path: Optional[str] = Field(default=None)
    max_samples: Optional[int] = Field(default=None)


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class ToxicityPredictResponse(BaseModel):
    """Response from toxicity prediction."""
    smiles: str
    is_valid: bool
    is_toxic: bool
    toxicity_probability: float
    label_probabilities: Dict[str, float] = {}


class ToxicityBatchPredictResponse(BaseModel):
    """Response from batch toxicity prediction."""
    predictions: List[ToxicityPredictResponse]
    total_predictions: int
    toxic_count: int
