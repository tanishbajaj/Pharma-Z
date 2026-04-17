"""
ESM-Enhanced DeepDTA Model

DeepDTA variant that replaces the CNN protein encoder with
pre-computed ESM-2 protein embeddings.

Architecture:
    Drug:    CNN encoder (same as baseline) → 96-dim
    Protein: Pre-computed ESM-2 embedding   → 1280-dim (650M model)
    Combined: 1376-dim → FC(1024) → FC(1024) → FC(512) → Linear(1)

This addresses the key finding from the DeepDTA paper:
    "CNN architecture failed to capture ordered relationships
     in amino acid sequences"
"""

import torch
import torch.nn as nn

from app.models.deep_dta_model import (
    DrugEncoder,
    DRUG_CHARSET_SIZE,
    BINDING_EMBEDDING_DIM,
    NUM_FILTERS,
)
from app.models.esm_protein_encoder import ESM_MODEL_CONFIGS


class DeepDTA_ESM(nn.Module):
    """
    ESM-Enhanced DeepDTA for binding affinity prediction.

    Drug encoding uses the same CNN as baseline DeepDTA.
    Protein encoding uses pre-computed ESM-2 embeddings (frozen, no training).
    The combined features are fed through FC layers for affinity prediction.

    Args:
        esm_model_name: ESM-2 model name to determine embedding dimension.
        drug_charset_size: Size of the drug character vocabulary.
        embedding_dim: Drug embedding dimension.
        num_filters: Number of CNN filters for drug encoder.
        fc_hidden_dim: Hidden dimension of FC classifier layers.
        dropout_rate: Dropout rate for regularization.
    """

    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        drug_charset_size: int = DRUG_CHARSET_SIZE,
        embedding_dim: int = BINDING_EMBEDDING_DIM,
        num_filters: int = NUM_FILTERS,
        fc_hidden_dim: int = 1024,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.esm_model_name = esm_model_name
        esm_config = ESM_MODEL_CONFIGS[esm_model_name]
        self.esm_embed_dim = esm_config["embed_dim"]

        # Drug encoder: same CNN as baseline DeepDTA
        self.drug_encoder = DrugEncoder(
            charset_size=drug_charset_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
        )

        # No protein encoder — we use pre-computed ESM embeddings

        # Projection layer to align ESM embedding dimension if needed
        drug_feature_dim = num_filters * 3  # 96

        # Combined: drug CNN features + ESM protein features
        combined_dim = drug_feature_dim + self.esm_embed_dim  # 96 + 1280 = 1376

        # Paper-style FC classifier: 1024 → 1024 → 512 → 1
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim // 2, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for FC and Conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        drug_tokens: torch.Tensor,
        protein_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict binding affinity for drug-protein pairs.

        Args:
            drug_tokens:        (batch_size, drug_max_length) integer indices
            protein_embeddings: (batch_size, esm_embed_dim) pre-computed ESM features

        Returns:
            binding_affinity: (batch_size, 1) predicted affinity scores
        """
        # Encode drug with CNN
        drug_features = self.drug_encoder(drug_tokens)  # (batch, 96)

        # Protein features are already pre-computed ESM embeddings
        # Just combine with drug features
        combined = torch.cat([drug_features, protein_embeddings], dim=1)  # (batch, 1376)
        binding_affinity = self.classifier(combined)  # (batch, 1)

        return binding_affinity
