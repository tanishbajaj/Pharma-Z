"""
Download Binding Affinity Datasets (KIBA & Davis)

Downloads data from the official DeepDTA GitHub repository
(https://github.com/hkmztrk/DeepDTA) and converts to CSV format.

The affinity matrix (Y) is stored as a Python 2 pickle containing a
numpy ndarray. We use encoding='latin1' for Python 3 compatibility.

Usage:
    python scripts/download_binding_data.py                 # Both datasets
    python scripts/download_binding_data.py --dataset kiba   # KIBA only
    python scripts/download_binding_data.py --dataset davis  # Davis only
"""

import argparse
import json
import os
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEPDTA_BASE_URL = (
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"
)

DATASETS = {
    "kiba": {
        "ligands_url": f"{DEEPDTA_BASE_URL}/kiba/ligands_can.txt",
        "proteins_url": f"{DEEPDTA_BASE_URL}/kiba/proteins.txt",
        "affinity_url": f"{DEEPDTA_BASE_URL}/kiba/Y",
        "folds_url_template": f"{DEEPDTA_BASE_URL}/kiba/folds/train_fold_setting1.txt",
        "test_fold_url": f"{DEEPDTA_BASE_URL}/kiba/folds/test_fold_setting1.txt",
        "transform": None,  # KIBA scores used directly
        "output_csv": "kiba.csv",
        "output_folds": "kiba_folds.json",
    },
    "davis": {
        "ligands_url": f"{DEEPDTA_BASE_URL}/davis/ligands_can.txt",
        "proteins_url": f"{DEEPDTA_BASE_URL}/davis/proteins.txt",
        "affinity_url": f"{DEEPDTA_BASE_URL}/davis/Y",
        "folds_url_template": f"{DEEPDTA_BASE_URL}/davis/folds/train_fold_setting1.txt",
        "test_fold_url": f"{DEEPDTA_BASE_URL}/davis/folds/test_fold_setting1.txt",
        "transform": "pKd",  # Convert Kd -> pKd
        "output_csv": "davis.csv",
        "output_folds": "davis_folds.json",
    },
}


# ---------------------------------------------------------------------------
# Download Helpers
# ---------------------------------------------------------------------------

def download_text(url: str) -> str:
    """Download a text file from a URL."""
    print(f"  Downloading: {url.split('/')[-1]}")
    req = urllib.request.Request(url, headers={"User-Agent": "Pharma-Z/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def download_binary(url: str) -> bytes:
    """Download a binary file from a URL."""
    print(f"  Downloading: {url.split('/')[-1]} (binary)")
    req = urllib.request.Request(url, headers={"User-Agent": "Pharma-Z/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    print(f"       ({len(data):,} bytes)")
    return data


def download_json(url: str) -> dict:
    """Download and parse a JSON file."""
    text = download_text(url)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------------

def process_dataset(dataset_name: str, output_dir: str) -> None:
    """
    Download and process a single dataset.
    
    Args:
        dataset_name: 'kiba' or 'davis'
        output_dir: Directory for output files
    """
    config = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"  Processing {dataset_name.upper()} Dataset")
    print(f"{'='*60}")

    # -------------------------------------------------------------------
    # 1) Download raw files
    # -------------------------------------------------------------------
    print("\n[1/4] Downloading raw data files...")

    ligands_dict = download_json(config["ligands_url"])
    proteins_dict = download_json(config["proteins_url"])

    # Y is a Python 2 pickle containing a numpy affinity matrix
    affinity_raw = download_binary(config["affinity_url"])

    print(f"       Drugs: {len(ligands_dict)}")
    print(f"       Proteins: {len(proteins_dict)}")

    # -------------------------------------------------------------------
    # 2) Parse affinity matrix (Python 2 pickle -> numpy array)
    # -------------------------------------------------------------------
    print("\n[2/4] Parsing affinity matrix...")

    # encoding='latin1' required for Python 2 pickles loaded in Python 3
    affinity_matrix = pickle.loads(affinity_raw, encoding="latin1")
    if not isinstance(affinity_matrix, np.ndarray):
        affinity_matrix = np.array(affinity_matrix, dtype=np.float64)

    drug_ids = list(ligands_dict.keys())
    protein_ids = list(proteins_dict.keys())

    print(f"       Affinity matrix shape: {affinity_matrix.shape}")
    print(f"       Expected: ({len(drug_ids)}, {len(protein_ids)})")

    assert affinity_matrix.shape == (len(drug_ids), len(protein_ids)), \
        f"Shape mismatch! Matrix {affinity_matrix.shape} vs drugs={len(drug_ids)} proteins={len(protein_ids)}"

    nan_count = int(np.sum(np.isnan(affinity_matrix)))
    valid_count = int(np.sum(~np.isnan(affinity_matrix) & ~np.isinf(affinity_matrix)))
    print(f"       Valid entries: {valid_count:,} / {affinity_matrix.size:,}")
    print(f"       Missing (NaN): {nan_count:,}")

    # -------------------------------------------------------------------
    # 3) Flatten to rows and apply affinity transform
    # -------------------------------------------------------------------
    print("\n[3/4] Flattening matrix to drug-target pairs...")

    rows = []
    for i, drug_id in enumerate(drug_ids):
        smiles = ligands_dict[drug_id]
        for j, protein_id in enumerate(protein_ids):
            affinity = affinity_matrix[i, j]

            # Skip NaN / missing values (inf in some datasets)
            if np.isnan(affinity) or np.isinf(affinity):
                continue

            # Apply transform if needed
            if config["transform"] == "pKd":
                # Davis: Kd (nM) -> pKd = -log10(Kd / 1e9)
                if affinity <= 0:
                    continue
                affinity = -np.log10(affinity / 1e9)

            protein_seq = proteins_dict[protein_id]
            rows.append({
                "smiles": smiles,
                "protein_sequence": protein_seq,
                "affinity_score": round(float(affinity), 6),
                "drug_id": drug_id,
                "protein_id": protein_id,
            })

    df = pd.DataFrame(rows)
    print(f"       Total pairs: {len(df):,}")
    print(f"       Affinity range: [{df['affinity_score'].min():.3f}, {df['affinity_score'].max():.3f}]")
    print(f"       Affinity mean: {df['affinity_score'].mean():.3f} +/- {df['affinity_score'].std():.3f}")
    print(f"       Unique drugs: {df['drug_id'].nunique()}")
    print(f"       Unique proteins: {df['protein_id'].nunique()}")
    print(f"       SMILES max length: {df['smiles'].str.len().max()}")
    print(f"       Protein max length: {df['protein_sequence'].str.len().max()}")

    # Save CSV
    csv_path = os.path.join(output_dir, config["output_csv"])
    df.to_csv(csv_path, index=False)
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"       Saved: {csv_path} ({file_size_mb:.1f} MB)")

    # -------------------------------------------------------------------
    # 4) Download fold indices
    # -------------------------------------------------------------------
    print("\n[4/4] Downloading fold indices...")

    try:
        train_folds_text = download_text(config["folds_url_template"])
        test_fold_text = download_text(config["test_fold_url"])

        # Parse fold indices (they are JSON arrays of indices)
        train_folds = json.loads(train_folds_text)
        test_fold = json.loads(test_fold_text)

        folds_data = {
            "train_folds": train_folds,
            "test_fold": test_fold,
            "dataset": dataset_name,
            "total_pairs": len(df),
        }

        folds_path = os.path.join(output_dir, config["output_folds"])
        with open(folds_path, "w") as f:
            json.dump(folds_data, f, indent=2)
        print(f"       Saved fold indices: {folds_path}")
        print(f"       Train folds: {len(train_folds)} folds")
        print(f"       Test fold size: {len(test_fold)} samples")
    except Exception as e:
        print(f"       WARNING: Could not download fold indices: {e}")
        print(f"       Training will use random splits instead.")

    print(f"\n  [DONE] {dataset_name.upper()} dataset ready!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download KIBA/Davis binding affinity datasets")
    parser.add_argument(
        "--dataset",
        choices=["kiba", "davis", "both"],
        default="both",
        help="Which dataset to download (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/raw/)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Navigate from scripts/ to data/raw/
        script_dir = Path(__file__).resolve().parent
        output_dir = str(script_dir.parent / "data" / "raw")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Pharma-Z: Binding Affinity Dataset Downloader")
    print("  Source: https://github.com/hkmztrk/DeepDTA")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    datasets_to_process = ["kiba", "davis"] if args.dataset == "both" else [args.dataset]

    for ds in datasets_to_process:
        process_dataset(ds, output_dir)

    print(f"\n{'='*60}")
    print("  All datasets downloaded successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
