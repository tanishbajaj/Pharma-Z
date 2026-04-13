"""
Streaming Graph Dataset — On-the-fly SMILES → Graph conversion

Streams molecular graphs from a SMILES file without loading the full
dataset into memory. Converts SMILES to PyG Data objects in chunks.

Enables training on the full ZINC 22.8M dataset on a single GPU.
"""

import os
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import IterableDataset

from app.models.graph_utils import (
    smiles_to_graph,
    NODE_FEATURE_DIM,
)
from app.utils.logging_config import logger


class StreamingGraphDataset(IterableDataset):
    """
    Streaming dataset that converts SMILES → molecular graphs on-the-fly.

    Instead of loading all graphs into RAM, reads the SMILES file in chunks
    and converts each chunk to PyG Data objects during iteration.

    Args:
        smiles_file:  Path to SMILES file (.csv or .smi).
        max_nodes:    Maximum atoms per molecule (skip larger ones).
        chunk_size:   Number of SMILES to read per chunk.
        max_samples:  Limit total molecules (None = use all).
        smiles_column: Column name for CSV files.
    """

    def __init__(
        self,
        smiles_file: str,
        max_nodes: int = 38,
        chunk_size: int = 10000,
        max_samples: Optional[int] = None,
        smiles_column: str = "smiles",
    ):
        super().__init__()
        self.smiles_file = smiles_file
        self.max_nodes = max_nodes
        self.chunk_size = chunk_size
        self.max_samples = max_samples
        self.smiles_column = smiles_column

        # Count total lines for progress reporting
        self._total_lines = self._count_lines()
        self._effective_total = min(
            self._total_lines, max_samples or float("inf")
        )

        logger.info(
            f"StreamingGraphDataset: {self._total_lines:,} molecules | "
            f"file={os.path.basename(smiles_file)} | "
            f"chunk_size={chunk_size:,} | max_nodes={max_nodes}"
        )

    def _count_lines(self) -> int:
        """Count data lines in the file (excluding header)."""
        count = 0
        with open(self.smiles_file, "r", encoding="utf-8", errors="ignore") as f:
            for i, _ in enumerate(f):
                count += 1
        # Subtract 1 for header
        return max(count - 1, 0)

    def __iter__(self):
        """
        Iterate through the file, yielding PyG Data objects.

        Reads SMILES in chunks, converts to graphs, yields one at a time.
        Each Data object includes padded_x, adj, bond_types for training.
        """
        ext = Path(self.smiles_file).suffix.lower()
        yielded = 0

        if ext == ".csv":
            yield from self._iter_csv()
        elif ext in (".smi", ".txt"):
            yield from self._iter_smi()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _iter_csv(self):
        """Stream from CSV file using pandas chunked reader."""
        import pandas as pd

        yielded = 0
        for chunk_df in pd.read_csv(
            self.smiles_file,
            chunksize=self.chunk_size,
            usecols=None,
        ):
            # Find SMILES column
            col = self.smiles_column
            if col not in chunk_df.columns:
                for candidate in ["smiles", "SMILES", "Smiles", "mol"]:
                    if candidate in chunk_df.columns:
                        col = candidate
                        break
                else:
                    col = chunk_df.columns[0]

            smiles_list = chunk_df[col].dropna().astype(str).tolist()

            for smiles in smiles_list:
                if self.max_samples and yielded >= self.max_samples:
                    return

                smiles = smiles.strip()
                if len(smiles) < 5 or smiles == "nan":
                    continue

                data = self._smiles_to_training_data(smiles)
                if data is not None:
                    yielded += 1
                    yield data

    def _iter_smi(self):
        """Stream from .smi/.txt file line by line."""
        yielded = 0
        with open(self.smiles_file, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f):
                if self.max_samples and yielded >= self.max_samples:
                    return

                line = line.strip()
                if not line:
                    continue

                # Skip header
                if line_num == 0 and (
                    "smiles" in line.lower() or "zinc" in line.lower()
                ):
                    continue

                smiles = line.split()[0] if line.split() else ""
                if len(smiles) < 5 or smiles == "nan":
                    continue

                data = self._smiles_to_training_data(smiles)
                if data is not None:
                    yielded += 1
                    yield data

    def _smiles_to_training_data(self, smiles: str):
        """
        Convert a SMILES string to a training-ready Data object.

        Adds padded_x (for decoder target) to the standard graph Data.
        Returns None if the molecule is invalid or too large.
        """
        graph = smiles_to_graph(smiles, max_nodes=self.max_nodes)
        if graph is None:
            return None

        # Add padded node features for decoder target
        padded_x = torch.zeros(
            self.max_nodes, NODE_FEATURE_DIM, dtype=torch.float32
        )
        n = graph.x.size(0)
        padded_x[:n] = graph.x
        graph.padded_x = padded_x

        return graph

    def __len__(self):
        """Approximate length (used for progress estimation)."""
        return int(self._effective_total)
