"""
DeepDTA Binding Affinity Prediction Model

1D-CNN based model for predicting drug-protein binding affinity.
Drug (SMILES) and protein (amino acid sequence) are independently
encoded via Conv1d networks, then combined via FC layers.

Reference: Öztürk et al., "DeepDTA: deep drug-target binding affinity prediction"
"""

import torch
import torch.nn as nn

from app.utils.constants import DEFAULT_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Constants for Binding Model
# ---------------------------------------------------------------------------

DRUG_MAX_LENGTH = 100       # Max SMILES length for drug
PROTEIN_MAX_LENGTH = 1000   # Max amino acid sequence length
DRUG_CHARSET_SIZE = 65      # Extended SMILES charset
PROTEIN_CHARSET_SIZE = 26   # 20 amino acids + padding + extras

NUM_FILTERS = 32
FILTER_LENGTHS = [4, 6, 8]  # Multi-scale 1D convolution kernels


class DrugEncoder(nn.Module):
    """
    1D-CNN encoder for drug SMILES strings.

    Embeds SMILES characters and applies multi-layer 1D convolution
    to extract local chemical substructure features.
    """

    def __init__(
        self,
        charset_size: int = DRUG_CHARSET_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_filters: int = NUM_FILTERS,
        max_length: int = DRUG_MAX_LENGTH,
    ):
        super().__init__()

        self.embedding = nn.Embedding(charset_size, embedding_dim, padding_idx=0)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=6, padding=3),
            nn.ReLU(),
            nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=8, padding=4),
            nn.ReLU(),
        )

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, drug_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_tokens: (batch_size, drug_max_length) integer indices.

        Returns:
            drug_features: (batch_size, num_filters * 3)
        """
        embedded = self.embedding(drug_tokens)          # (batch, seq, embed)
        embedded = embedded.permute(0, 2, 1)            # (batch, embed, seq) for Conv1d

        conv_output = self.conv_layers(embedded)        # (batch, num_filters*3, seq')
        pooled = self.global_max_pool(conv_output)      # (batch, num_filters*3, 1)
        features = pooled.squeeze(-1)                   # (batch, num_filters*3)

        return features


class ProteinEncoder(nn.Module):
    """
    1D-CNN encoder for protein amino acid sequences.

    Same architecture as DrugEncoder but with larger input dimensions
    suited for protein sequences.
    """

    def __init__(
        self,
        charset_size: int = PROTEIN_CHARSET_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_filters: int = NUM_FILTERS,
        max_length: int = PROTEIN_MAX_LENGTH,
    ):
        super().__init__()

        self.embedding = nn.Embedding(charset_size, embedding_dim, padding_idx=0)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=12, padding=6),
            nn.ReLU(),
        )

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, protein_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein_tokens: (batch_size, protein_max_length) integer indices.

        Returns:
            protein_features: (batch_size, num_filters * 3)
        """
        embedded = self.embedding(protein_tokens)
        embedded = embedded.permute(0, 2, 1)

        conv_output = self.conv_layers(embedded)
        pooled = self.global_max_pool(conv_output)
        features = pooled.squeeze(-1)

        return features


class DeepDTA(nn.Module):
    """
    DeepDTA: Drug-Target Binding Affinity Prediction.

    Combines drug and protein CNN encoders, concatenates their features,
    and predicts a continuous binding affinity score via FC layers.
    """

    def __init__(
        self,
        drug_charset_size: int = DRUG_CHARSET_SIZE,
        protein_charset_size: int = PROTEIN_CHARSET_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_filters: int = NUM_FILTERS,
        fc_hidden_dim: int = 1024,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.drug_encoder = DrugEncoder(
            charset_size=drug_charset_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
        )

        self.protein_encoder = ProteinEncoder(
            charset_size=protein_charset_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
        )

        # Combined feature size = drug (num_filters*3) + protein (num_filters*3)
        combined_dim = num_filters * 3 * 2

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim // 2, 1),  # Single output: binding affinity
        )

    def forward(
        self,
        drug_tokens: torch.Tensor,
        protein_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict binding affinity for drug-protein pairs.

        Args:
            drug_tokens:    (batch_size, drug_max_length)
            protein_tokens: (batch_size, protein_max_length)

        Returns:
            binding_affinity: (batch_size, 1) predicted affinity scores.
        """
        drug_features = self.drug_encoder(drug_tokens)       # (batch, num_filters*3)
        protein_features = self.protein_encoder(protein_tokens)  # (batch, num_filters*3)

        combined = torch.cat([drug_features, protein_features], dim=1)  # (batch, combined_dim)
        binding_affinity = self.classifier(combined)  # (batch, 1)

        return binding_affinity


# ---------------------------------------------------------------------------
# Sequence Encoding Utilities
# ---------------------------------------------------------------------------

# SMILES character set for drug encoding
DRUG_CHAR_TO_INDEX = {}
_drug_chars = (
    " #%()+-./0123456789=@ABCFHIKLMNOPSV"
    "Z[\\]abcegilnoprstu"
)
for i, ch in enumerate(_drug_chars, start=1):  # 0 = padding
    DRUG_CHAR_TO_INDEX[ch] = i

# Amino acid character set for protein encoding
AMINO_ACID_CHARS = "ACDEFGHIKLMNPQRSTVWY"
PROTEIN_CHAR_TO_INDEX = {aa: i + 1 for i, aa in enumerate(AMINO_ACID_CHARS)}


def encode_drug_sequence(smiles: str, max_length: int = DRUG_MAX_LENGTH) -> list:
    """
    Encode a SMILES string into integer indices for the drug encoder.

    Args:
        smiles:     SMILES string.
        max_length: Maximum sequence length (will be padded/truncated).

    Returns:
        List of integer indices.
    """
    encoded = []
    for char in smiles[:max_length]:
        encoded.append(DRUG_CHAR_TO_INDEX.get(char, 0))

    # Pad to max_length
    encoded.extend([0] * (max_length - len(encoded)))
    return encoded


def encode_protein_sequence(sequence: str, max_length: int = PROTEIN_MAX_LENGTH) -> list:
    """
    Encode a protein amino acid sequence into integer indices.

    Args:
        sequence:   Amino acid sequence string.
        max_length: Maximum sequence length (will be padded/truncated).

    Returns:
        List of integer indices.
    """
    encoded = []
    for aa in sequence[:max_length]:
        encoded.append(PROTEIN_CHAR_TO_INDEX.get(aa.upper(), 0))

    encoded.extend([0] * (max_length - len(encoded)))
    return encoded
