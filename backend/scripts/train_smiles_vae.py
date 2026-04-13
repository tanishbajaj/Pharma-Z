"""
Phase 2: SMILES VAE Training — Standalone Script

Train the SMILES Variational Autoencoder on ZINC data, then evaluate with
full MOSES benchmark metrics (Polykovskiy et al., 2020).

Usage:
    cd backend

    # Quick test (1K molecules, 3 epochs)
    python scripts/train_smiles_vae.py --max-samples 1000 --epochs 3

    # ZINC250K baseline (recommended first run)
    python scripts/train_smiles_vae.py --smiles-file data/raw/zinc250k.csv --epochs 50

    # ZINC1M full training with SMILES augmentation
    python scripts/train_smiles_vae.py --smiles-file data/raw/zinc_1m.csv --epochs 50 --augment

    # Cyclical KL annealing + temperature sampling
    python scripts/train_smiles_vae.py --epochs 100 --kl-anneal cyclical --temperature 0.8

    # Resume from checkpoint
    python scripts/train_smiles_vae.py --resume data/saved_models/smiles_vae/vae_epoch_20.pt --epochs 50
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from app.services.molecule_service import (
    load_zinc_dataset,
    prepare_training_data,
    initialize_vae_model,
    train_smiles_vae,
    generate_molecules,
    evaluate_generated_molecules,
    get_tokenizer,
)
from app.models.smiles_streaming_dataset import create_streaming_dataloader
from app.models.training_utils import get_training_device, load_model_checkpoint
from app.utils.constants import (
    DEFAULT_LATENT_DIM,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
)

# Auto-switch to streaming mode for datasets larger than this
STREAMING_THRESHOLD = 500_000  # molecules


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: SMILES VAE Baseline Training"
    )

    # Data
    parser.add_argument(
        "--smiles-file", type=str, default=None,
        help="Path to SMILES CSV file (default: data/raw/zinc250k.csv)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit training samples (for quick testing)"
    )

    # Model
    parser.add_argument(
        "--latent-dim", type=int, default=DEFAULT_LATENT_DIM,
        help="Latent space dimension"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM,
        help="Token embedding dimension"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM,
        help="LSTM hidden dimension"
    )

    # Training
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4, per MOSES benchmark)"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--kl-weight", type=float, default=0.5,
        help="KL divergence weight / beta (default: 0.5, MOSES uses 1.0)"
    )
    parser.add_argument(
        "--kl-anneal", type=str, default="linear",
        choices=["linear", "cyclical"],
        help="KL annealing strategy: linear (default) or cyclical (TGVAE)"
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable SMILES randomization (data augmentation, Bjerrum 2017)"
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )

    # Evaluation
    parser.add_argument(
        "--eval-samples", type=int, default=30000,
        help="Number of molecules to generate for evaluation (MOSES: 30K)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (0.8=focused/valid [default], 1.0=standard, 1.2=creative)"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip post-training evaluation"
    )

    args = parser.parse_args()

    # ===================================================================
    # Banner
    # ===================================================================
    print("=" * 65)
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║        Pharma-Z  ·  Phase 2: SMILES VAE           ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print("=" * 65)

    device = get_training_device()
    print(f"\n  Device:        {device}")
    if device.type == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM:          {vram:.1f} GB")

    print(f"\n  Latent dim:    {args.latent_dim}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Hidden dim:    {args.hidden_dim}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  LR:            {args.lr}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  KL weight:     {args.kl_weight}")
    print(f"  KL anneal:     {args.kl_anneal}")
    print(f"  Augment:       {args.augment}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Eval samples:  {args.eval_samples:,}")
    if args.max_samples:
        print(f"  Max samples:   {args.max_samples}")
    if args.resume:
        print(f"  Resume from:   {args.resume}")

    # ===================================================================
    # Step 1: Count dataset size & decide streaming vs in-memory
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 1/4] Loading ZINC dataset")
    print(f"{'=' * 65}")

    smiles_file = args.smiles_file or os.path.join("data", "raw", "zinc250k.csv")
    tokenizer = get_tokenizer()

    # Count lines quickly to decide loading strategy
    with open(smiles_file, "r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f) - 1  # subtract header
    print(f"  ✓ Dataset: {os.path.basename(smiles_file)} ({total_lines:,} molecules)")

    use_streaming = (not args.max_samples) and (total_lines > STREAMING_THRESHOLD)
    print(f"  ✓ Mode: {'STREAMING (chunked, low RAM)' if use_streaming else 'IN-MEMORY'}")

    # ===================================================================
    # Step 2: Prepare DataLoader
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 2/4] Preparing DataLoader")
    print(f"{'=' * 65}")

    t1 = time.time()
    print(f"  ✓ Vocabulary size: {tokenizer.vocabulary_size}")
    smiles_list = None  # only populated in in-memory mode for novelty eval

    if use_streaming:
        data_loader, estimated_batches = create_streaming_dataloader(
            file_path=smiles_file,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            chunk_size=50_000,
            num_workers=0,  # Must be 0 — nohup/background processes deadlock with workers
            augment_smiles=args.augment,
        )
        print(f"  ✓ Streaming DataLoader: ~{estimated_batches:,} batches/epoch ({time.time()-t1:.1f}s)")
        if args.augment:
            print(f"  ✓ SMILES randomization: ON (encoder sees random, decoder targets canonical)")
    else:
        t0 = time.time()
        smiles_list = load_zinc_dataset(
            file_path=smiles_file,
            max_samples=args.max_samples,
        )
        print(f"  ✓ Loaded {len(smiles_list):,} SMILES in {time.time()-t0:.1f}s")
        data_loader = prepare_training_data(
            smiles_list,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Must be 0 — nohup/background processes deadlock with workers
        )
        print(f"  ✓ DataLoader ready: {len(data_loader)} batches ({time.time()-t1:.1f}s)")

    # ===================================================================
    # Step 3: Train
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 3/4] Training SMILES VAE")
    print(f"{'=' * 65}\n")

    model = initialize_vae_model(
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"  Loading checkpoint: {args.resume}")
        load_model_checkpoint(model, args.resume)

    t_train = time.time()
    history = train_smiles_vae(
        data_loader=data_loader,
        model=model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        save_every=args.save_every,
        kl_anneal_mode=args.kl_anneal,
    )
    train_time = time.time() - t_train

    print(f"\n  ✓ Training complete in {train_time:.1f}s")
    print(f"  ✓ Final loss: {history['total_loss'][-1]:.4f}")
    print(f"    Recon: {history['reconstruction_loss'][-1]:.4f}")
    print(f"    KL:    {history['kl_loss'][-1]:.4f}")

    # ===================================================================
    # Step 4: Evaluate — Full MOSES Benchmark Metrics
    # ===================================================================
    if not args.skip_eval:
        print(f"\n{'=' * 65}")
        print("[Step 4/4] Evaluating generation quality (MOSES metrics)")
        print(f"{'=' * 65}")

        print(f"\n  Generating {args.eval_samples:,} molecules (temp={args.temperature})...")
        results = generate_molecules(
            num_molecules=args.eval_samples,
            model=model,
            temperature=args.temperature,
        )

        generated_smiles = [r["smiles"] for r in results]

        # In streaming mode smiles_list is None — load a 50K reference sample
        # for Novelty, Fragment Sim, Scaffold Sim, and SNN.
        # 50K molecules ≈ 50MB RAM — safe on any machine.
        eval_reference = smiles_list
        if eval_reference is None:
            print("  Loading 50K reference sample from training file for Novelty/Frag/Scaff/SNN...")
            eval_reference = load_zinc_dataset(
                file_path=smiles_file,
                max_samples=50_000,
            )
            print(f"  ✓ Reference sample loaded: {len(eval_reference):,} molecules")

        print("  Computing MOSES metrics: Validity, Uniqueness, Novelty,")
        print("    IntDiv1/2, Fragment Sim, Scaffold Sim, SNN...")
        metrics = evaluate_generated_molecules(
            generated_smiles=generated_smiles,
            training_smiles=eval_reference,
        )

        # ── MOSES Metrics Table ──
        print(f"\n  ┌──────────────────────────────────────────────────┐")
        print(f"  │        SMILES VAE — MOSES Evaluation Metrics      │")
        print(f"  ├──────────────────────────────────────────────────┤")
        print(f"  │  Generated:     {metrics['total_generated']:>8,}                      │")
        print(f"  │  Valid:         {metrics['valid_count']:>8,}  ({metrics['validity_rate']:.1%})             │")
        print(f"  │  Unique:        {metrics['unique_count']:>8,}  ({metrics['uniqueness_rate']:.1%})             │")
        print(f"  ├──────────────────────────────────────────────────┤")
        print(f"  │  Novelty:              {metrics['novelty_rate']:.4f}                  │")
        print(f"  │  IntDiv1:              {metrics['intdiv1']:.4f}  (target ≥0.85)   │")
        print(f"  │  IntDiv2:              {metrics['intdiv2']:.4f}                  │")
        print(f"  ├──────────────────────────────────────────────────┤")
        print(f"  │  Fragment Sim:         {metrics['frag_similarity']:.4f}  (target ≥0.99)   │")
        print(f"  │  Scaffold Sim:         {metrics['scaff_similarity']:.4f}  (target ≥0.90)   │")
        print(f"  │  SNN:                  {metrics['snn']:.4f}  (target ≥0.55)   │")
        print(f"  └──────────────────────────────────────────────────┘")

        # MOSES benchmark reference
        print(f"\n  Reference (MOSES VAE baseline):")
        print(f"    Validity=97.7%  Unique@1k=100%  Novelty=69.5%")
        print(f"    IntDiv1=0.856   Frag=0.999      Scaff=0.939  SNN=0.626")

        # Show sample generated molecules
        valid_mols = [r for r in results if r["is_valid"]]
        if valid_mols:
            print(f"\n  Sample valid molecules ({len(valid_mols):,} total valid):")
            for i, mol in enumerate(valid_mols[:10]):
                props = mol.get("properties", {})
                qed = props.get("qed_score", 0)
                mw = props.get("molecular_weight", 0)
                logp = props.get("logp", 0)
                print(
                    f"    {i+1:>2}. {mol['smiles'][:55]:<55s} "
                    f"MW={mw:>6.1f}  logP={logp:>5.2f}  QED={qed:.3f}"
                )

        # Save metrics
        metrics_path = os.path.join("data", "saved_models", "smiles_vae", "eval_metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓ Metrics saved: {metrics_path}")

    # ===================================================================
    # Done
    # ===================================================================
    print(f"\n{'=' * 65}")
    print(f"  ✓ Phase 2 Complete — SMILES VAE Baseline Trained!")
    print(f"{'=' * 65}")
    print(f"\n  Checkpoints: data/saved_models/smiles_vae/")
    print(f"  To generate via API:")
    print(f"    uvicorn app.main:app --reload")
    print(f"    POST /api/v1/molecules/generate")
    print(f"\n  Next: Phase 3 — GraphVAE Training")
    print(f"    python scripts/train_graph_vae.py --smiles-file data/raw/zinc_1m.csv --epochs 50")


if __name__ == "__main__":
    main()
