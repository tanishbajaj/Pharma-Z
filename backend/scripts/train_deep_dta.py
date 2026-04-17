"""
DeepDTA Training Script — Binding Affinity Prediction

Standalone training script for the DeepDTA binding affinity model.
Supports both baseline (CNN) and ESM-enhanced model variants.

Usage:
    cd backend

    # Quick smoke test (CPU, ~2 min)
    python scripts/train_deep_dta.py --dataset davis --max-samples 500 --epochs 3

    # Davis baseline (GPU, 2-4 hours)
    python scripts/train_deep_dta.py --dataset davis --epochs 100 --batch-size 256

    # KIBA baseline (GPU, 6-12 hours)
    python scripts/train_deep_dta.py --dataset kiba --epochs 100 --batch-size 256

    # ESM-enhanced (pre-computes ESM-2 embeddings, then trains)
    python scripts/train_deep_dta.py --dataset kiba --model-type esm --epochs 100

    # Cold-drug split (research evaluation)
    python scripts/train_deep_dta.py --dataset kiba --split-mode cold-drug

    # Resume training from checkpoint
    python scripts/train_deep_dta.py --dataset kiba --resume data/saved_models/deep_dta/best_model.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from app.services.binding_service import (
    load_binding_dataset,
    create_train_val_test_split,
    prepare_binding_data,
    initialize_dta_model,
    train_deep_dta,
    evaluate_model,
)
from app.models.binding_metrics import compute_all_metrics
from app.models.training_utils import (
    get_training_device,
    load_model_checkpoint,
)


# ---------------------------------------------------------------------------
# Paper benchmark targets (for comparison)
# ---------------------------------------------------------------------------

PAPER_BENCHMARKS = {
    "davis": {"ci": 0.878, "mse": 0.261},
    "kiba": {"ci": 0.863, "mse": 0.194},
}


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner():
    print("=" * 65)
    print("  +=====================================================+")
    print("  |     Pharma-Z  -  Phase 4: DeepDTA Binding Affinity  |")
    print("  +=====================================================+")
    print("=" * 65)


# ---------------------------------------------------------------------------
# ESM Embedding Pre-computation
# ---------------------------------------------------------------------------

def precompute_esm_embeddings(
    df,
    dataset_name: str,
    esm_model_name: str,
    cache_dir: str,
) -> dict:
    """
    Pre-compute ESM-2 embeddings for all unique proteins in the dataset.
    Results are cached to disk for reuse.

    Returns:
        Dict mapping protein_sequence -> embedding tensor
    """
    from app.models.esm_protein_encoder import ESMProteinEncoder

    cache_path = os.path.join(cache_dir, f"{dataset_name}_{esm_model_name}_embeddings.pt")

    # Try loading from cache first
    if os.path.exists(cache_path):
        print(f"\n  [CACHE] Loading cached ESM embeddings from: {cache_path}")
        return ESMProteinEncoder.load_cache(cache_path)

    # Compute fresh embeddings
    print(f"\n  [ESM] Pre-computing ESM-2 embeddings (one-time operation)...")
    unique_proteins = df["protein_sequence"].unique()
    print(f"     Unique proteins: {len(unique_proteins)}")

    # Create protein_id -> sequence mapping
    # Use sequence itself as key for lookup during training
    protein_dict = {f"P{i}": seq for i, seq in enumerate(unique_proteins)}

    encoder = ESMProteinEncoder(
        model_name=esm_model_name,
        device=torch.device("cpu"),  # CPU avoids competing with other GPU users
    )
    # batch_size=1, max_length=512 keeps CPU RAM usage safe
    embeddings_by_id = encoder.compute_embeddings(
        protein_dict, batch_size=1, max_length=512
    )

    # Re-key by sequence for easy lookup during training
    embeddings_by_seq = {}
    for pid, seq in protein_dict.items():
        embeddings_by_seq[seq] = embeddings_by_id[pid]

    # Cache to disk
    encoder.save_cache(embeddings_by_seq, cache_path)
    encoder.unload_model()

    return embeddings_by_seq


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def run_training(args):
    """Main training pipeline."""
    print_banner()

    # ------- Device Detection -------
    device = get_training_device()
    print(f"\n  Device:        {device}")
    if device.type == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM:          {vram:.1f} GB")

    print(f"\n  Dataset:       {args.dataset.upper()}")
    print(f"  Model type:    {args.model_type}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  LR:            {args.lr}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Patience:      {args.patience}")
    print(f"  Split mode:    {args.split_mode}")
    if args.max_samples:
        print(f"  Max samples:   {args.max_samples}")
    if args.model_type == "esm":
        print(f"  ESM model:     {args.esm_model}")
    if args.resume:
        print(f"  Resume from:   {args.resume}")

    # ===================================================================
    # Step 1: Load Dataset
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 1/7] Loading Dataset")
    print(f"{'=' * 65}")

    start_time = time.time()
    df = load_binding_dataset(args.dataset, max_samples=args.max_samples)
    print(f"  Loaded {len(df):,} drug-target pairs")
    print(f"  Unique drugs: {df['smiles'].nunique()}")
    print(f"  Unique proteins: {df['protein_sequence'].nunique()}")
    print(f"  Affinity range: [{df['affinity_score'].min():.3f}, {df['affinity_score'].max():.3f}]")
    print()

    # ------- Step 2: ESM Embeddings (if ESM mode) -------
    protein_embeddings = None
    if args.model_type == "esm":
        print(f"\n{'=' * 65}")
        print("[Step 2/7] ESM-2 Protein Embeddings")
        print(f"{'=' * 65}")

        cache_dir = os.path.join(
            Path(__file__).resolve().parent.parent, "data", "esm_cache"
        )
        protein_embeddings = precompute_esm_embeddings(
            df, args.dataset, args.esm_model, cache_dir
        )
        print()

    # ===================================================================
    # Step 3: Split Data
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 3/7] Creating Train/Val/Test Split")
    print(f"{'=' * 65}")

    folds_path = None
    if args.split_mode == "fold":
        folds_path = os.path.join(
            Path(__file__).resolve().parent.parent,
            "data", "raw", f"{args.dataset}_folds.json"
        )

    train_df, val_df, test_df = create_train_val_test_split(
        df,
        split_mode=args.split_mode,
        fold_num=args.fold_num,
        folds_path=folds_path,
        seed=args.seed,
    )
    print(f"  Train: {len(train_df):,} pairs")
    print(f"  Val:   {len(val_df):,} pairs")
    print(f"  Test:  {len(test_df):,} pairs")
    print()

    # ===================================================================
    # Step 4: Prepare DataLoaders
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 4/7] Preparing DataLoaders")
    print(f"{'=' * 65}")

    train_loader = prepare_binding_data(
        train_df["smiles"].tolist(),
        train_df["protein_sequence"].tolist(),
        train_df["affinity_score"].tolist(),
        batch_size=args.batch_size,
        shuffle=True,
        protein_embeddings=protein_embeddings,
    )

    val_loader = prepare_binding_data(
        val_df["smiles"].tolist(),
        val_df["protein_sequence"].tolist(),
        val_df["affinity_score"].tolist(),
        batch_size=args.batch_size,
        shuffle=False,
        protein_embeddings=protein_embeddings,
    )

    test_loader = prepare_binding_data(
        test_df["smiles"].tolist(),
        test_df["protein_sequence"].tolist(),
        test_df["affinity_score"].tolist(),
        batch_size=args.batch_size,
        shuffle=False,
        protein_embeddings=protein_embeddings,
    )
    print()

    # ===================================================================
    # Step 5: Initialize Model
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 5/7] Initializing Model")
    print(f"{'=' * 65}")

    model_kwargs = {}
    if args.model_type == "esm":
        model_kwargs["esm_model_name"] = args.esm_model

    model = initialize_dta_model(model_type=args.model_type, **model_kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        load_model_checkpoint(model, args.resume)

    print()

    # ===================================================================
    # Step 6: Training
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 6/7] Training")
    print(f"{'=' * 65}")

    save_dir = os.path.join(
        Path(__file__).resolve().parent.parent,
        "data", "saved_models", "deep_dta"
    )
    if args.model_type == "esm":
        save_dir = os.path.join(
            Path(__file__).resolve().parent.parent,
            "data", "saved_models", "deep_dta_esm"
        )

    training_history = train_deep_dta(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=save_dir,
        patience=args.patience,
    )

    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time/60:.1f} minutes")
    print()

    # ------- Step 7: Final Evaluation -------
    if not args.skip_eval:
        print(f"\n{'=' * 65}")
        print("[Step 7/7] Final Test Set Evaluation")
        print(f"{'=' * 65}")

        test_metrics = evaluate_model(model, test_loader, device)

        # Get paper benchmarks for comparison
        paper = PAPER_BENCHMARKS.get(args.dataset, {})

        print()
        print("  ┌──────────────────────────────────────────────────┐")
        print(f"  │     DeepDTA — Test Set Evaluation ({args.dataset.upper()})          │")
        print("  ├──────────────────────────────────────────────────┤")
        print(f"  │  Model:         {args.model_type:>30}   │")
        print(f"  │  Test pairs:    {len(test_df):>30,}   │")
        print("  ├──────────────────────────────────────────────────┤")

        # MSE
        mse_str = f"{test_metrics['mse']:.4f}"
        if "mse" in paper:
            delta = test_metrics["mse"] - paper["mse"]
            mse_str += f"  (paper: {paper['mse']:.3f})"
        print(f"  │  MSE:           {mse_str:>30}   │")

        # RMSE
        print(f"  │  RMSE:          {test_metrics['rmse']:>30.4f}   │")

        # CI
        ci_str = f"{test_metrics['ci']:.4f}"
        if "ci" in paper:
            delta = test_metrics["ci"] - paper["ci"]
            sign = "+" if delta >= 0 else ""
            ci_str += f"  (paper: {paper['ci']:.3f}, {sign}{delta:.3f})"
        print(f"  │  CI:            {ci_str:>30}   │")

        # Pearson r
        print(f"  │  Pearson r:     {test_metrics['pearson_r']:>30.4f}   │")
        print("  └──────────────────────────────────────────────────┘")
        print()

        # ------- Save Results -------
        results = {
            "dataset": args.dataset,
            "model_type": args.model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "split_mode": args.split_mode,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "test_metrics": test_metrics,
            "training_time_minutes": round(train_time / 60, 2),
            "paper_benchmarks": paper,
            "training_history": {
                k: [float(v) for v in vals]
                for k, vals in training_history.items()
            },
        }

        results_path = os.path.join(save_dir, "training_results.json")
        os.makedirs(save_dir, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved: {results_path}")

    print()
    print("  [DONE] DeepDTA training pipeline complete!")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train DeepDTA Binding Affinity Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", choices=["kiba", "davis"], default="davis",
                        help="Dataset to train on (default: davis)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")

    # Model
    parser.add_argument("--model-type", choices=["baseline", "esm"], default="baseline",
                        help="Model variant (default: baseline)")
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D",
                        help="ESM-2 model name (default: esm2_t33_650M_UR50D)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256, paper spec)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001, paper spec)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20, 0=disabled)")

    # Splitting
    parser.add_argument("--split-mode", choices=["random", "cold-drug", "cold-protein", "fold"],
                        default="random", help="Data split strategy (default: random)")
    parser.add_argument("--fold-num", type=int, default=0,
                        help="Fold number for fold split mode (0-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Misc
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip final test set evaluation")

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
