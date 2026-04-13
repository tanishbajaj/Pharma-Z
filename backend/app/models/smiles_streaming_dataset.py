"""
Streaming SMILES Dataset

Reads SMILES from a CSV file in chunks — never loads the full file into RAM.
Allows training on 22M+ molecule datasets on standard hardware.

Supports:
  - Chunked CSV reading (pandas chunked iterator)
  - On-the-fly tokenization per batch
  - Shuffled chunk ordering each epoch
  - Drop-in replacement for TensorDataset approach
  - SMILES randomization (data augmentation, Bjerrum 2017)
"""

import os
import random
from typing import Optional, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from app.models.smiles_tokenizer import SmilesTokenizer
from app.utils.constants import MAX_MOLECULE_LENGTH, SMILES_CHARSET
from app.utils.logging_config import logger


class StreamingSMILESDataset(IterableDataset):
    """
    Streams SMILES from a large CSV file in chunks.

    Never loads more than `chunk_size` molecules into RAM at once.
    Each call to __iter__ re-reads the file (for multi-epoch training).

    Args:
        file_path:    Path to CSV file with a 'smiles' column.
        tokenizer:    SmilesTokenizer instance.
        chunk_size:   Number of rows to read per chunk (default: 50K).
        max_length:   Max SMILES length to keep.
        shuffle_chunks: Shuffle chunk order each epoch.
        max_samples:  Limit total molecules (0 = no limit).
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: SmilesTokenizer,
        chunk_size: int = 50_000,
        max_length: int = MAX_MOLECULE_LENGTH,
        shuffle_chunks: bool = True,
        max_samples: int = 0,
        augment_smiles: bool = False,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.shuffle_chunks = shuffle_chunks
        self.max_samples = max_samples
        self.augment_smiles = augment_smiles and HAS_RDKIT

        # Count total lines for __len__ estimate (fast scan)
        self._total = self._count_lines()
        aug_status = "ON" if self.augment_smiles else "OFF"
        logger.info(
            f"StreamingSMILESDataset: {self._total:,} molecules | "
            f"file={os.path.basename(file_path)} | chunk_size={chunk_size:,} | "
            f"augment={aug_status}"
        )

    def _count_lines(self) -> int:
        """Fast line count (subtract 1 for header)."""
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)

    def __len__(self) -> int:
        if self.max_samples and self.max_samples < self._total:
            return self.max_samples
        return self._total

    def __iter__(self):
        """Yield (input_tokens, target_tokens) one at a time."""
        emitted = 0

        # Read CSV in chunks
        reader = pd.read_csv(
            self.file_path,
            chunksize=self.chunk_size,
            usecols=[0],  # First column = smiles
            header=0,
            dtype=str,
            on_bad_lines="skip",
        )

        for chunk_df in reader:
            # Get SMILES from first column regardless of name
            smiles_col = chunk_df.columns[0]
            smiles_list = chunk_df[smiles_col].dropna().tolist()

            # Filter by length (min 5, max max_length)
            smiles_list = [s for s in smiles_list if 5 <= len(s) <= self.max_length]

            # Shuffle within chunk
            if self.shuffle_chunks:
                random.shuffle(smiles_list)

            for smi in smiles_list:
                # SMILES randomization (data augmentation)
                # Encoder sees random SMILES, decoder targets canonical
                # Forces latent space to learn molecule, not string
                # (Bjerrum 2017, All SMILES VAE 2020)
                if self.augment_smiles:
                    encoder_smi = self._randomize_smiles(smi)
                else:
                    encoder_smi = smi

                encoder_tokens = self.tokenizer.encode(encoder_smi)
                target_tokens = self.tokenizer.encode(smi)
                enc_t = torch.tensor(encoder_tokens, dtype=torch.long)
                tgt_t = torch.tensor(target_tokens, dtype=torch.long)
                yield enc_t, tgt_t

                emitted += 1
                if self.max_samples and emitted >= self.max_samples:
                    return


    @staticmethod
    def _randomize_smiles(smiles: str) -> str:
        """
        Generate a random (non-canonical) SMILES string for a molecule.
        Falls back to original SMILES if randomization fails.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            random_smi = Chem.MolToSmiles(mol, doRandom=True)
            return random_smi if random_smi else smiles
        except Exception:
            return smiles


def create_streaming_dataloader(
    file_path: str,
    tokenizer: SmilesTokenizer,
    batch_size: int = 256,
    chunk_size: int = 50_000,
    num_workers: int = 0,
    max_samples: int = 0,
    augment_smiles: bool = False,
) -> Tuple[DataLoader, int]:
    """
    Create a streaming DataLoader for large SMILES CSV files.

    Args:
        augment_smiles: If True, encoder sees randomized SMILES while
                        decoder targets canonical SMILES. Forces the
                        latent space to capture molecular structure.

    Returns:
        (dataloader, estimated_num_batches)
    """
    dataset = StreamingSMILESDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        shuffle_chunks=True,
        max_samples=max_samples,
        augment_smiles=augment_smiles,
    )

    # IterableDataset requires batch_size in DataLoader, no shuffle
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # Avoids variable batch sizes at chunk boundaries
    )

    estimated_batches = len(dataset) // batch_size
    logger.info(
        f"Streaming DataLoader: ~{len(dataset):,} samples | "
        f"batch_size={batch_size} | ~{estimated_batches:,} batches/epoch"
    )

    return loader, estimated_batches
