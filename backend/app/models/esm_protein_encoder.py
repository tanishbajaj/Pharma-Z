"""
ESM-2 Protein Encoder for Binding Affinity Prediction

Uses Meta's pretrained ESM-2 (Evolutionary Scale Modeling) protein language
model to generate rich protein representations, replacing the weak CNN
protein encoder from the original DeepDTA paper.

Reference:
  Lin et al., "Evolutionary-scale prediction of atomic-level protein
  structure with a language model", Science, 2023.

Why ESM-2:
  - Trained on 250 million protein sequences via masked language modeling
  - Captures evolutionary conservation, structural contacts, functional motifs
  - DeepDTA paper admitted CNN fails for proteins: "CNN architecture failed
    to capture ordered relationships in amino acid sequences"
  - ESM-2 embeddings encode far richer information than simple CNN features

Usage:
    encoder = ESMProteinEncoder(model_name="esm2_t33_650M_UR50D")
    embeddings = encoder.compute_embeddings({"P1": "MKTVRQ...", "P2": "ACDEF..."})
    encoder.save_cache(embeddings, "data/esm_cache/kiba_proteins.pt")

    # During training — load cached embeddings (no ESM model needed)
    embeddings = ESMProteinEncoder.load_cache("data/esm_cache/kiba_proteins.pt")
"""

import os
from typing import Dict, Optional

import torch
import numpy as np


# ESM-2 model configurations with output dimensions
ESM_MODEL_CONFIGS = {
    "esm2_t6_8M_UR50D": {"layers": 6, "embed_dim": 320, "params": "8M"},
    "esm2_t12_35M_UR50D": {"layers": 12, "embed_dim": 480, "params": "35M"},
    "esm2_t30_150M_UR50D": {"layers": 30, "embed_dim": 640, "params": "150M"},
    "esm2_t33_650M_UR50D": {"layers": 33, "embed_dim": 1280, "params": "650M"},
    "esm2_t36_3B_UR50D": {"layers": 36, "embed_dim": 2560, "params": "3B"},
}


class ESMProteinEncoder:
    """
    Pre-computes ESM-2 embeddings for protein sequences.

    Strategy: Compute embeddings ONCE for all unique proteins in a dataset,
    save to disk, then load during training. This avoids running ESM-2
    during every training epoch — zero overhead after pre-computation.

    Attributes:
        model_name: ESM-2 model identifier (e.g., "esm2_t33_650M_UR50D")
        embed_dim:  Output embedding dimension (e.g., 1280 for 650M model)
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.config = ESM_MODEL_CONFIGS[model_name]
        self.embed_dim = self.config["embed_dim"]
        self.num_layers = self.config["layers"]
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._model = None
        self._alphabet = None
        self._batch_converter = None

    def _load_model(self):
        """Lazily load the ESM-2 model (downloads weights on first use)."""
        if self._model is not None:
            return

        try:
            import esm
        except ImportError:
            raise ImportError(
                "ESM package not installed. Install with: pip install fair-esm\n"
                "See: https://github.com/facebookresearch/esm"
            )

        print(f"  Loading ESM-2 model: {self.model_name} ({self.config['params']} params)...")
        model_loader = getattr(esm.pretrained, self.model_name)
        self._model, self._alphabet = model_loader()
        self._batch_converter = self._alphabet.get_batch_converter()
        self._model = self._model.eval().to(self.device)

        total_params = sum(p.numel() for p in self._model.parameters())
        print(f"  ESM-2 loaded: {total_params:,} parameters on {self.device}")
        print(f"  Output embedding dim: {self.embed_dim}")

    def compute_embeddings(
        self,
        protein_sequences: Dict[str, str],
        batch_size: int = 4,
        max_length: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ESM-2 embeddings for a dict of protein sequences.

        Args:
            protein_sequences: Dict of {protein_id: amino_acid_sequence}
            batch_size: Number of proteins to process at once
            max_length: Truncate proteins longer than this (ESM-2 limit ~1024)

        Returns:
            Dict of {protein_id: embedding_tensor(embed_dim,)}
        """
        self._load_model()

        embeddings = {}
        protein_items = list(protein_sequences.items())
        total = len(protein_items)

        print(f"  Computing ESM-2 embeddings for {total} proteins...")

        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_items = protein_items[start_idx:end_idx]

            # Prepare batch: list of (label, sequence) tuples
            # Truncate long sequences to avoid OOM
            data = [
                (pid, seq[:max_length]) for pid, seq in batch_items
            ]

            batch_labels, batch_strs, batch_tokens = self._batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            # Get per-residue representations from the last layer
            with torch.no_grad():
                results = self._model(
                    batch_tokens,
                    repr_layers=[self.num_layers],
                    return_contacts=False,
                )

            token_representations = results["representations"][self.num_layers]

            # Mean-pool over residue positions (excluding BOS/EOS tokens)
            batch_lens = (batch_tokens != self._alphabet.padding_idx).sum(1)

            for i, (pid, _) in enumerate(batch_items):
                # Tokens: [BOS, residue_1, residue_2, ..., residue_N, EOS, PAD, ...]
                # We average over residue_1 to residue_N (indices 1 to tokens_len-1)
                tokens_len = batch_lens[i]
                embedding = token_representations[i, 1:tokens_len - 1].mean(0)
                embeddings[pid] = embedding.cpu()

            processed = min(end_idx, total)
            print(f"    Processed: {processed}/{total} proteins", end="\r")

        print(f"    Processed: {total}/{total} proteins — Done!")
        return embeddings

    def save_cache(self, embeddings: Dict[str, torch.Tensor], cache_path: str):
        """Save computed embeddings to disk."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(embeddings, cache_path)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"  Saved ESM embeddings cache: {cache_path} ({size_mb:.1f} MB)")

    @staticmethod
    def load_cache(cache_path: str) -> Dict[str, torch.Tensor]:
        """Load pre-computed embeddings from disk."""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"ESM embedding cache not found: {cache_path}\n"
                f"Run the training script with --model-type esm to generate it."
            )
        embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  Loaded ESM embeddings: {len(embeddings)} proteins from {cache_path}")
        return embeddings

    def unload_model(self):
        """Free GPU memory after pre-computation is done."""
        if self._model is not None:
            del self._model
            del self._alphabet
            del self._batch_converter
            self._model = None
            self._alphabet = None
            self._batch_converter = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  ESM-2 model unloaded from memory.")
