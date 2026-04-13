"""
Molecule Schemas

Pydantic request/response models for molecule generation endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class MoleculeGenerateRequest(BaseModel):
    """Request to generate novel molecules."""
    num_molecules: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of molecules to generate",
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to trained model checkpoint (uses latest if None)",
    )


class MoleculeValidateRequest(BaseModel):
    """Request to validate a SMILES string."""
    smiles: str = Field(
        ...,
        description="SMILES string to validate",
        examples=["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"],
    )


class MoleculeTrainRequest(BaseModel):
    """Request to start VAE training."""
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to SMILES CSV dataset (defaults to ZINC250K)",
    )
    max_samples: Optional[int] = Field(
        default=None,
        description="Limit number of training samples (for quick testing)",
    )
    num_epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=64, ge=8, le=512)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    latent_dim: int = Field(default=128, ge=16, le=512)
    kl_weight: float = Field(default=0.1, ge=0, le=10.0)


class MoleculeEvaluateRequest(BaseModel):
    """Request to evaluate generated molecules."""
    generated_smiles: List[str] = Field(
        ...,
        description="List of generated SMILES to evaluate",
    )
    training_smiles: Optional[List[str]] = Field(
        default=None,
        description="Training SMILES for novelty calculation",
    )


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class MoleculeProperties(BaseModel):
    """Computed molecular properties."""
    molecular_weight: float = 0.0
    logp: float = 0.0
    num_h_donors: int = 0
    num_h_acceptors: int = 0
    tpsa: float = 0.0
    num_rotatable_bonds: int = 0
    num_rings: int = 0
    qed_score: float = 0.0


class GeneratedMolecule(BaseModel):
    """A single generated molecule with validation info."""
    smiles: str
    is_valid: bool
    properties: Dict[str, Any] = {}


class MoleculeGenerateResponse(BaseModel):
    """Response from molecule generation."""
    molecules: List[GeneratedMolecule]
    total_generated: int
    valid_count: int
    validity_rate: float


class MoleculeValidateResponse(BaseModel):
    """Response from SMILES validation."""
    input_smiles: str
    is_valid: bool
    canonical_smiles: Optional[str] = None
    properties: Dict[str, Any] = {}


class TrainingStatusResponse(BaseModel):
    """Response after starting/completing training."""
    status: str
    message: str
    epochs_completed: int = 0
    final_loss: Optional[float] = None
    training_history: Optional[Dict[str, List[float]]] = None


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for generated molecules."""
    total_generated: int
    valid_count: int
    unique_count: int
    validity_rate: float
    uniqueness_rate: float
    novelty_rate: float
    diversity_score: float
