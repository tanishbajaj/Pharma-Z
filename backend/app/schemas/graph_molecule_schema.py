"""
Graph Molecule Schemas — Pydantic models for GraphVAE API endpoints
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class GraphTrainRequest(BaseModel):
    """Request body for POST /molecules/train-graph"""
    smiles_file: Optional[str] = Field(
        None,
        description="Path to SMILES file. Auto-detected if not provided.",
    )
    max_samples: Optional[int] = Field(
        None,
        description="Limit training samples (for quick testing).",
    )
    max_nodes: int = Field(38, description="Maximum atoms per molecule.")
    latent_dim: int = Field(128, description="Latent space dimension.")
    hidden_dim: int = Field(256, description="Hidden layer dimension.")
    batch_size: int = Field(128, description="Training batch size.")
    learning_rate: float = Field(0.001, description="Adam learning rate.")
    num_epochs: int = Field(50, description="Number of training epochs.")
    kl_weight: float = Field(0.005, description="KL divergence weight (β).")


class GraphTrainResponse(BaseModel):
    """Response for POST /molecules/train-graph"""
    status: str
    dataset_size: int
    num_epochs: int
    final_loss: float
    final_adj_loss: float
    final_node_loss: float
    final_kl_loss: float
    training_time_seconds: float
    checkpoint_dir: str


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class GraphGenerateRequest(BaseModel):
    """Request body for POST /molecules/generate-graph"""
    num_molecules: int = Field(10, ge=1, le=1000, description="Number to generate.")
    temperature: float = Field(1.0, ge=0.1, le=3.0, description="Sampling temperature.")
    adj_threshold: float = Field(0.5, ge=0.1, le=0.9, description="Adjacency threshold.")
    checkpoint_path: Optional[str] = Field(None, description="Model checkpoint path.")


class GeneratedMolecule(BaseModel):
    """A single generated molecule."""
    smiles: Optional[str]
    is_valid: bool
    properties: Dict[str, Any] = {}


class GraphGenerateResponse(BaseModel):
    """Response for POST /molecules/generate-graph"""
    num_generated: int
    num_valid: int
    validity_rate: float
    molecules: List[GeneratedMolecule]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class GraphEvalRequest(BaseModel):
    """Request body for POST /molecules/evaluate-graph"""
    num_samples: int = Field(100, ge=10, le=10000, description="Molecules to generate for eval.")
    temperature: float = Field(1.0, ge=0.1, le=3.0, description="Sampling temperature.")


class GraphEvalResponse(BaseModel):
    """Response for POST /molecules/evaluate-graph"""
    num_generated: int
    num_valid: int
    num_unique: int
    validity: float
    uniqueness: float
    novelty: float
    sample_molecules: List[str]
