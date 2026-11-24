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
        shuffle_buffer: int = 50_000,
    ):
        super().__init__()
        self.smiles_file = smiles_file
        self.max_nodes = max_nodes
        self.chunk_size = chunk_size
        self.max_samples = max_samples
        self.smiles_column = smiles_column
        # DataLoader cannot shuffle an IterableDataset, so we shuffle here with
        # a reservoir buffer. Without it, batches follow ZINC's tranche order and
        # are chemically correlated, which is poison for VAE training.
        self.shuffle_buffer = shuffle_buffer

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

    def _shard(self):
        """
        This worker's (id, count).

        Every DataLoader worker gets its own copy of the dataset and would
        otherwise replay the whole file, so N workers would train on N copies
        of the data. Sharding by worker id makes each worker take a disjoint
        stride instead.
        """
        info = torch.utils.data.get_worker_info()
        if info is None:
            return 0, 1
        return info.id, info.num_workers

    def __iter__(self):
        """
        Iterate through the file, yielding PyG Data objects.

        Shards across DataLoader workers, converts SMILES to graphs, and
        shuffles through a reservoir buffer before yielding.
        """
        ext = Path(self.smiles_file).suffix.lower()
        if ext == ".csv":
            raw = self._iter_csv()
        elif ext in (".smi", ".txt"):
            raw = self._iter_smi()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        graphs = self._to_graphs(raw)
        yield from self._shuffled(graphs)

    def _to_graphs(self, smiles_iterator):
        for smiles in smiles_iterator:
            data = self._smiles_to_training_data(smiles)
            if data is not None:
                yield data

    def _shuffled(self, iterator):
        """
        Reservoir shuffle over a streaming iterator.

        Fills a buffer, then for each new item yields a random element from
        the buffer and replaces it. torch.initial_seed() differs per worker
        AND per epoch, so the order changes every pass without extra wiring.
        """
        if self.shuffle_buffer <= 1:
            yield from iterator
            return

        import random
        rng = random.Random(torch.initial_seed() & 0xFFFFFFFF)

        buffer = []
        for item in iterator:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(item)
            else:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = item

        rng.shuffle(buffer)
        yield from buffer

    def _iter_csv(self):
        """Stream SMILES strings from a CSV, sharded across workers."""
        import pandas as pd

        worker_id, num_workers = self._shard()
        seen = 0  # global molecule index, before sharding

        for chunk_df in pd.read_csv(
            self.smiles_file,
            chunksize=self.chunk_size,
            usecols=None,
        ):
            col = self.smiles_column
            if col not in chunk_df.columns:
                for candidate in ["smiles", "SMILES", "Smiles", "mol"]:
                    if candidate in chunk_df.columns:
                        col = candidate
                        break
                else:
                    col = chunk_df.columns[0]

            for smiles in chunk_df[col].dropna().astype(str).tolist():
                if self.max_samples and seen >= self.max_samples:
                    return
                index = seen
                seen += 1
                if index % num_workers != worker_id:
                    continue

                smiles = smiles.strip()
                if len(smiles) < 5 or smiles == "nan":
                    continue
                yield smiles

    def _iter_smi(self):
        """Stream SMILES strings from a .smi/.txt file, sharded across workers."""
        worker_id, num_workers = self._shard()
        seen = 0

        with open(self.smiles_file, "r", encoding="utf-8", errors="ignore") as handle:
            for line_num, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                if line_num == 0 and (
                    "smiles" in line.lower() or "zinc" in line.lower()
                ):
                    continue

                if self.max_samples and seen >= self.max_samples:
                    return
                index = seen
                seen += 1
                if index % num_workers != worker_id:
                    continue

                parts = line.split()
                smiles = parts[0] if parts else ""
                if len(smiles) < 5 or smiles == "nan":
                    continue
                yield smiles

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
