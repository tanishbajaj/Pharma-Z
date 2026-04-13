"""
Prepare Full Training CSV — Validates, cleans, and exports the ZINC dataset.

Reads zinc_merged.smi, validates SMILES with RDKit, canonicalizes,
filters by length, and writes a clean CSV ready for GPU training.

Usage:
    cd backend
    python scripts/prepare_full_dataset.py                       # Full 22M
    python scripts/prepare_full_dataset.py --max-molecules 5000000  # 5M subset
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings for speed
RDLogger.DisableLog("rdApp.*")

INPUT_FILE = os.path.join("data", "raw", "zinc_merged.smi")
OUTPUT_DIR = os.path.join("data", "raw")


def fmt(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def main():
    parser = argparse.ArgumentParser(description="Prepare full training CSV")
    parser.add_argument("--input", type=str, default=INPUT_FILE)
    parser.add_argument("--max-molecules", type=int, default=0,
                        help="Max molecules (0 = all)")
    parser.add_argument("--max-length", type=int, default=120,
                        help="Max SMILES string length")
    parser.add_argument("--min-length", type=int, default=5,
                        help="Min SMILES string length")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip RDKit validation (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pharma-Z — Full Dataset Preparation")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Read raw SMILES
    # ---------------------------------------------------------------
    print(f"\n[Step 1/3] Reading: {args.input}")
    t0 = time.time()

    raw_smiles = []
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()  # skip header
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if s:
                raw_smiles.append(s)
            if args.max_molecules and len(raw_smiles) >= args.max_molecules:
                break

    print(f"  ✓ Read {len(raw_smiles):,} raw SMILES ({time.time()-t0:.1f}s)")

    # ---------------------------------------------------------------
    # Step 2: Validate, canonicalize, deduplicate, filter
    # ---------------------------------------------------------------
    print(f"\n[Step 2/3] Validating & cleaning")
    t1 = time.time()

    valid_smiles = []
    seen = set()
    invalid = 0
    too_long = 0
    too_short = 0
    duplicates = 0

    for i, smi in enumerate(raw_smiles):
        # Length filter first (fast)
        if len(smi) > args.max_length:
            too_long += 1
            continue
        if len(smi) < args.min_length:
            too_short += 1
            continue

        if args.skip_validation:
            canonical = smi
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid += 1
                continue
            canonical = Chem.MolToSmiles(mol)

        if canonical in seen:
            duplicates += 1
            continue

        seen.add(canonical)
        valid_smiles.append(canonical)

        if (i + 1) % 1_000_000 == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            sys.stdout.write(
                f"\r    Processed: {i+1:>12,} | "
                f"Valid: {len(valid_smiles):>10,} | "
                f"Rate: {rate:,.0f}/s"
            )
            sys.stdout.flush()

    elapsed = time.time() - t1
    print(f"\r    Processed: {len(raw_smiles):>12,} in {elapsed:.1f}s")
    print(f"\n  Results:")
    print(f"    Valid & unique: {len(valid_smiles):>12,}")
    print(f"    Invalid:        {invalid:>12,}")
    print(f"    Too long (>{args.max_length}): {too_long:>12,}")
    print(f"    Too short (<{args.min_length}): {too_short:>12,}")
    print(f"    Duplicates:     {duplicates:>12,}")

    # ---------------------------------------------------------------
    # Step 3: Write output CSVs
    # ---------------------------------------------------------------
    print(f"\n[Step 3/3] Writing training CSVs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full dataset
    full_path = os.path.join(OUTPUT_DIR, "zinc_full.csv")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("smiles\n")
        for s in valid_smiles:
            f.write(f"{s}\n")
    full_size = os.path.getsize(full_path) / 1024 / 1024
    print(f"  ✓ zinc_full.csv:  {len(valid_smiles):>12,} molecules ({full_size:.1f} MB)")

    # 5M subset
    if len(valid_smiles) > 5_000_000:
        sub5m = valid_smiles[:5_000_000]
        path_5m = os.path.join(OUTPUT_DIR, "zinc_5m.csv")
        with open(path_5m, "w", encoding="utf-8") as f:
            f.write("smiles\n")
            for s in sub5m:
                f.write(f"{s}\n")
        print(f"  ✓ zinc_5m.csv:    {len(sub5m):>12,} molecules ({os.path.getsize(path_5m)/1024/1024:.1f} MB)")

    # 1M subset
    sub1m = valid_smiles[:1_000_000]
    path_1m = os.path.join(OUTPUT_DIR, "zinc_1m.csv")
    with open(path_1m, "w", encoding="utf-8") as f:
        f.write("smiles\n")
        for s in sub1m:
            f.write(f"{s}\n")
    print(f"  ✓ zinc_1m.csv:    {len(sub1m):>12,} molecules ({os.path.getsize(path_1m)/1024/1024:.1f} MB)")

    # 250K subset
    sub250k = valid_smiles[:250_000]
    path_250k = os.path.join(OUTPUT_DIR, "zinc250k.csv")
    with open(path_250k, "w", encoding="utf-8") as f:
        f.write("smiles\n")
        for s in sub250k:
            f.write(f"{s}\n")
    print(f"  ✓ zinc250k.csv:   {len(sub250k):>12,} molecules ({os.path.getsize(path_250k)/1024/1024:.1f} MB)")

    # Stats
    lengths = [len(s) for s in valid_smiles]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)

    print(f"\n  Dataset stats:")
    print(f"    Avg SMILES length: {avg_len:.1f}")
    print(f"    Min length:        {min_len}")
    print(f"    Max length:        {max_len}")

    print(f"\n{'=' * 60}")
    print(f"  ✓ Done! Full dataset ready for GPU training.")
    print(f"{'=' * 60}")
    print(f"\n  Training commands:")
    print(f"    # Quick baseline")
    print(f"    python scripts/train_smiles_vae.py --smiles-file data/raw/zinc250k.csv --epochs 30")
    print(f"    # Full training")
    print(f"    python scripts/train_smiles_vae.py --smiles-file data/raw/zinc_full.csv --epochs 50 --batch-size 256")


if __name__ == "__main__":
    main()
