"""
Pipeline Schemas

Pydantic models for the full drug discovery pipeline endpoint.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    """Request to run the full discovery pipeline."""
    smiles: str = Field(..., description="Drug candidate SMILES")
    protein_sequence: str = Field(..., description="Target protein sequence")
    num_generated: int = Field(
        default=0, ge=0, le=100,
        description="Number of additional molecules to generate and rank (0 = skip)",
    )


class PipelineBatchRequest(BaseModel):
    """Request to run pipeline for multiple drug candidates."""
    smiles_list: List[str] = Field(..., description="List of drug candidate SMILES")
    protein_sequence: str = Field(..., description="Target protein sequence")


class PipelineResultResponse(BaseModel):
    """Response from the pipeline."""
    status: str
    smiles: str
    canonical_smiles: Optional[str] = None
    protein_sequence: Optional[str] = None
    binding_affinity: float = 0.0
    toxicity_probability: float = 0.0
    is_toxic: bool = False
    drug_likeness: float = 0.0
    final_score: float = 0.0
    properties: Dict[str, Any] = {}
    ranking_weights: Dict[str, float] = {}
    ranked_candidates: Optional[List[Dict[str, Any]]] = None
    total_candidates: Optional[int] = None
    message: Optional[str] = None


class PipelineBatchResponse(BaseModel):
    """Response from batch pipeline."""
    candidates: List[Dict[str, Any]]
    total_ranked: int
