"""
ZINC 2D SMILES Downloader — Parses .wget file & downloads tranch files

Parses the ZINC-downloader-2D-smi.wget file (stored in data/),
downloads .smi tranch files in parallel using curl.exe (follows redirects),
deduplicates SMILES, and merges into training-ready CSVs.

This script was used to build the Pharma-Z ZINC dataset from
http://files.docking.org/2D/ tranches.

Usage:
    cd backend

    # Download all tranches (no molecule limit)
    python scripts/download_zinc_wget.py --max-molecules 0 --workers 10

    # Download with 1M molecule limit
    python scripts/download_zinc_wget.py --max-molecules 1000000 --workers 8

    # Skip download, just re-merge existing files
    python scripts/download_zinc_wget.py --skip-download
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WGET_FILE = Path(__file__).resolve().parent.parent / "data" / "ZINC-downloader-2D-smi.wget"
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DOWNLOAD_DIR = BASE_DIR / "zinc_2d_download"
MERGED_SMI = BASE_DIR / "zinc_merged.smi"
TRAINING_CSV = BASE_DIR / "zinc250k.csv"
FULL_CSV = BASE_DIR / "zinc_1m.csv"


def fmt(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Step 1: Parse .wget file
# ---------------------------------------------------------------------------

def parse_wget_file(path: Path) -> list:
    """
    Parse lines like:
        mkdir -pv BA && wget http://files.docking.org/2D/BA/BAAA.smi -O BA/BAAA.smi
    Returns list of dicts: {url, output_path}
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Extract URL
            url_match = re.search(r"(https?://\S+\.smi)", line)
            if not url_match:
                continue
            url = url_match.group(1)

            # Extract -O output path
            out_match = re.search(r"-O\s+(\S+)", line)
            output_path = out_match.group(1) if out_match else url.split("/")[-1]

            entries.append({"url": url, "output_path": output_path})

    return entries


# ---------------------------------------------------------------------------
# Step 2: Download single file
# ---------------------------------------------------------------------------

def download_one(entry: dict, dest_dir: Path) -> tuple:
    """
    Download a single .smi file using curl.exe.
    Returns (success: bool, file_size: int, smiles_count: int).
    """
    dest = dest_dir / entry["output_path"]

    # Skip if already exists with content
    if dest.exists() and dest.stat().st_size > 100:
        lines = sum(1 for _ in open(dest, "r", encoding="utf-8", errors="ignore"))
        return True, dest.stat().st_size, lines

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "curl.exe", "-sL",
                "--retry", "2",
                "--retry-delay", "1",
                "--connect-timeout", "15",
                "--max-time", "60",
                "-o", str(dest),
                entry["url"],
            ],
            capture_output=True, text=True, timeout=90,
        )

        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 50:
            lines = sum(1 for _ in open(dest, "r", encoding="utf-8", errors="ignore"))
            return True, dest.stat().st_size, max(0, lines - 1)  # minus header
        else:
            if dest.exists():
                dest.unlink()
            return False, 0, 0

    except Exception:
        if dest.exists():
            dest.unlink()
        return False, 0, 0


# ---------------------------------------------------------------------------
# Step 3: Batch download with limits
# ---------------------------------------------------------------------------

def download_batch(
    entries: list,
    max_files: int = None,
    limit_bytes: int = None,
    max_molecules: int = None,
    workers: int = 8,
) -> dict:
    """Download files with size/count limits. Uses parallel workers."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if max_files:
        entries = entries[:max_files]

    total_size = 0
    total_molecules = 0
    downloaded = 0
    failed = 0

    print(f"  Starting parallel download ({workers} workers, {len(entries)} files)...")

    # Use ThreadPoolExecutor for parallel downloads
    batch_size = min(workers * 4, len(entries))

    for batch_start in range(0, len(entries), batch_size):
        batch_entries = entries[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_one, entry, DOWNLOAD_DIR): entry
                for entry in batch_entries
            }

            for future in as_completed(futures):
                success, fsize, mol_count = future.result()
                if success:
                    total_size += fsize
                    total_molecules += mol_count
                    downloaded += 1
                else:
                    failed += 1

        # Print progress
        done = batch_start + len(batch_entries)
        sys.stdout.write(
            f"\r    [{done:>5}/{len(entries)}] "
            f"Size: {fmt(total_size):>10} | "
            f"Molecules: {total_molecules:>10,} | "
            f"OK: {downloaded} | Failed: {failed}   "
        )
        sys.stdout.flush()

        # Check limits
        if limit_bytes and total_size >= limit_bytes:
            print(f"\n  ✓ Size limit reached: {fmt(total_size)}")
            break
        if max_molecules and total_molecules >= max_molecules:
            print(f"\n  ✓ Molecule limit reached: {total_molecules:,}")
            break

    print(f"\n\n  Download summary:")
    print(f"    Downloaded: {downloaded}")
    print(f"    Failed:     {failed}")
    print(f"    Total size: {fmt(total_size)}")
    print(f"    Molecules:  ~{total_molecules:,}")

    return {
        "downloaded": downloaded,
        "failed": failed,
        "total_size": total_size,
        "total_molecules": total_molecules,
    }


# ---------------------------------------------------------------------------
# Step 4: Merge all .smi files → CSV
# ---------------------------------------------------------------------------

def merge_and_create_csv(max_molecules: int = None) -> int:
    """
    Merge all downloaded .smi files into training CSVs.

    Creates:
        zinc_merged.smi  — All unique SMILES (one per line)
        zinc250k.csv     — First 250K molecules
        zinc_1m.csv      — First 1M molecules
    """
    smi_files = sorted(DOWNLOAD_DIR.rglob("*.smi"))
    if not smi_files:
        print("  ✗ No .smi files found!")
        return 0

    print(f"  Processing {len(smi_files)} .smi files...")

    all_smiles = []
    seen = set()
    processed_files = 0

    for smi_file in smi_files:
        try:
            with open(smi_file, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    # Skip headers
                    if line_num == 0 and ("smiles" in line.lower() or "zinc" in line.lower()):
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    smiles = parts[0]
                    # Basic validation: must contain atoms
                    if len(smiles) < 3 or not any(c.isalpha() for c in smiles):
                        continue

                    if smiles not in seen:
                        seen.add(smiles)
                        all_smiles.append(smiles)

                        if max_molecules and len(all_smiles) >= max_molecules:
                            break

            processed_files += 1
            if processed_files % 200 == 0:
                sys.stdout.write(
                    f"\r    Files: {processed_files}/{len(smi_files)} | "
                    f"Unique SMILES: {len(all_smiles):,}"
                )
                sys.stdout.flush()

        except Exception:
            continue

        if max_molecules and len(all_smiles) >= max_molecules:
            break

    print(f"\r    Files: {processed_files}/{len(smi_files)} | "
          f"Unique SMILES: {len(all_smiles):,}                   ")

    # Write merged .smi
    print(f"\n  Writing merged file: {MERGED_SMI.name}")
    with open(MERGED_SMI, "w", encoding="utf-8") as f:
        f.write("smiles\n")
        for s in all_smiles:
            f.write(f"{s}\n")
    print(f"  ✓ {MERGED_SMI.name}: {len(all_smiles):,} molecules ({fmt(MERGED_SMI.stat().st_size)})")

    # Write zinc250k.csv (first 250K)
    subset_250k = all_smiles[:250_000]
    print(f"  Writing: {TRAINING_CSV.name}")
    with open(TRAINING_CSV, "w", encoding="utf-8") as f:
        f.write("smiles\n")
        for s in subset_250k:
            f.write(f"{s}\n")
    print(f"  ✓ {TRAINING_CSV.name}: {len(subset_250k):,} molecules ({fmt(TRAINING_CSV.stat().st_size)})")

    # Write zinc_1m.csv if we have > 250K
    if len(all_smiles) > 250_000:
        subset_1m = all_smiles[:1_000_000]
        print(f"  Writing: {FULL_CSV.name}")
        with open(FULL_CSV, "w", encoding="utf-8") as f:
            f.write("smiles\n")
            for s in subset_1m:
                f.write(f"{s}\n")
        print(f"  ✓ {FULL_CSV.name}: {len(subset_1m):,} molecules ({fmt(FULL_CSV.stat().st_size)})")

    return len(all_smiles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ZINC 2D SMI Downloader")
    parser.add_argument("--limit-gb", type=float, default=0,
                        help="Max download size in GB (0 = no limit)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max number of tranch files to download (0 = all)")
    parser.add_argument("--max-molecules", type=int, default=1_000_000,
                        help="Max unique molecules to collect (default: 1M, 0 = unlimited)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download workers")
    parser.add_argument("--wget-file", type=str, default=str(WGET_FILE))
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, just merge existing files")
    args = parser.parse_args()

    limit_bytes = int(args.limit_gb * 1024**3) if args.limit_gb > 0 else None

    print("=" * 70)
    print("  Pharma-Z — ZINC 2D SMILES Downloader")
    print("=" * 70)

    wget_path = Path(args.wget_file)
    if not wget_path.exists():
        print(f"  ✗ File not found: {wget_path}")
        sys.exit(1)

    # Step 1: Parse
    print(f"\n[Step 1/3] Parsing wget file: {wget_path.name}")
    entries = parse_wget_file(wget_path)
    print(f"  ✓ Found {len(entries)} download entries")

    # Step 2: Download
    if not args.skip_download:
        print(f"\n[Step 2/3] Downloading ZINC 2D SMILES tranches")
        if limit_bytes:
            print(f"  Size limit: {fmt(limit_bytes)}")
        if args.max_files:
            print(f"  File limit: {args.max_files}")
        if args.max_molecules:
            print(f"  Molecule target: {args.max_molecules:,}")
        else:
            print(f"  Molecule target: unlimited")
        print(f"  Workers: {args.workers}")
        print(f"  Output: {DOWNLOAD_DIR}")
        print()

        download_batch(
            entries,
            max_files=args.max_files or None,
            limit_bytes=limit_bytes,
            max_molecules=args.max_molecules or None,
            workers=args.workers,
        )
    else:
        print(f"\n[Step 2/3] Skipping download (--skip-download)")

    # Step 3: Merge
    print(f"\n[Step 3/3] Merging & creating training CSVs")
    total = merge_and_create_csv(max_molecules=args.max_molecules or None)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Done! {total:,} unique molecules ready for training.")
    print(f"{'=' * 70}")
    print(f"\n  Files in {BASE_DIR}:")
    for f in sorted(BASE_DIR.iterdir()):
        if f.is_file() and f.name != ".gitkeep":
            print(f"    • {f.name} ({fmt(f.stat().st_size)})")

    print(f"\n  Next steps:")
    print(f"    # Validate & create full CSV")
    print(f"    python scripts/prepare_full_dataset.py")
    print(f"    # Train SMILES VAE")
    print(f"    python scripts/train_smiles_vae.py --smiles-file data/raw/zinc_1m.csv --epochs 50")


if __name__ == "__main__":
    main()
