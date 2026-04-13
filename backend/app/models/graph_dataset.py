"""
Graph Molecule Dataset — PyTorch Geometric Dataset for ZINC SMILES

Loads SMILES from .smi or .csv files, converts to molecular graphs,
and provides a torch_geometric InMemoryDataset for efficient batching.

Supports:
    - Lazy loading for large datasets (>1M molecules)
    - Caching processed graphs to disk
    - Configurable max_nodes filter
"""

import os
from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset

from app.models.graph_utils import (
    smiles_to_graph,
    NODE_FEATURE_DIM,
)
from app.utils.logging_config import logger


class ZINCGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for ZINC molecular graphs.

    Reads SMILES from a file, converts each valid molecule to a graph
    Data object, and caches the result for fast reloading.

    Args:
        root:          Root directory for processed data cache.
        smiles_file:   Path to the SMILES file (.smi or .csv).
        max_nodes:     Maximum atoms per molecule (skip larger ones).
        max_samples:   Limit the number of molecules loaded (for testing).
        smiles_column: Column name if the file is a CSV.
        transform:     Optional transform applied per sample.
        pre_transform: Optional transform applied during processing.
    """

    def __init__(
        self,
        root: str,
        smiles_file: str,
        max_nodes: int = 38,
        max_samples: Optional[int] = None,
        smiles_column: str = "smiles",
        transform=None,
        pre_transform=None,
    ):
        self.smiles_file = smiles_file
        self.max_nodes = max_nodes
        self.max_samples = max_samples
        self.smiles_column = smiles_column

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(self.smiles_file)]

    @property
    def processed_file_names(self) -> List[str]:
        # Include max_nodes and max_samples in filename for cache busting
        suffix = f"_n{self.max_nodes}"
        if self.max_samples:
            suffix += f"_s{self.max_samples}"
        return [f"zinc_graphs{suffix}.pt"]

    def process(self):
        """Load SMILES, convert to graphs, save to disk."""
        logger.info(f"Processing SMILES file: {self.smiles_file}")
        logger.info(f"Max nodes: {self.max_nodes}, Max samples: {self.max_samples}")

        smiles_list = self._load_smiles()
        logger.info(f"Loaded {len(smiles_list):,} SMILES strings")

        # Convert to graphs with progress
        data_list: List[Data] = []
        skipped = 0

        for i, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles, max_nodes=self.max_nodes)
            if graph is not None:
                # Add padded node features for the decoder target
                padded_x = torch.zeros(
                    self.max_nodes, NODE_FEATURE_DIM, dtype=torch.float32
                )
                n = graph.x.size(0)
                padded_x[:n] = graph.x
                graph.padded_x = padded_x

                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)

                data_list.append(graph)
            else:
                skipped += 1

            if (i + 1) % 50000 == 0:
                logger.info(
                    f"  Processed {i + 1:,}/{len(smiles_list):,} | "
                    f"Valid: {len(data_list):,} | Skipped: {skipped:,}"
                )

        logger.info(
            f"Processing complete: {len(data_list):,} valid graphs, "
            f"{skipped:,} skipped"
        )

        self.save(data_list, self.processed_paths[0])

    def _load_smiles(self) -> List[str]:
        """Load SMILES strings from .smi or .csv file."""
        file_path = self.smiles_file
        ext = Path(file_path).suffix.lower()

        smiles_list: List[str] = []

        if ext == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            # Find the SMILES column
            col = self.smiles_column
            if col not in df.columns:
                for candidate in ["smiles", "SMILES", "Smiles", "mol"]:
                    if candidate in df.columns:
                        col = candidate
                        break
                else:
                    col = df.columns[0]
            smiles_list = df[col].dropna().astype(str).tolist()

        elif ext in (".smi", ".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    # Skip header
                    if line_num == 0 and ("smiles" in line.lower() or "zinc" in line.lower()):
                        continue
                    # First column is SMILES
                    parts = line.split()
                    if parts:
                        smiles_list.append(parts[0])
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        if self.max_samples is not None:
            smiles_list = smiles_list[:self.max_samples]

        return smiles_list
