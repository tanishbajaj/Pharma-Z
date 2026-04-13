"""
Phase 2: RGCVAE-Inspired GraphVAE Training — Standalone Script

Train the enhanced Graph Variational Autoencoder on ZINC molecular graphs,
then evaluate with validity, uniqueness, novelty, and diversity metrics.

Architecture enhancements (from 5 research papers):
    - RGIN encoder with per-bond-type MLPs (RGCVAE, Rigoni et al. 2025)
    - Rich 28-dim atom features (type + valence + charge + chirality + hybrid)
    - Cyclical β annealing (TGVAE, Nguyen & Karolak 2024)
    - Cross-entropy node loss (GraphVAE, Simonovsky & Komodakis 2018)
    - global_add_pool (consensus from papers 1, 4, 5)
    - Valence-masked molecule generation (RGCVAE + JT-VAE)
    - Symmetric decoder with implicit node probabilities
    - Weighted reconstruction loss (λ_A, λ_F, λ_E)
    - MST connectivity enforcement at generation time

Usage:
    cd backend

    # Quick test (1K molecules, 5 epochs)
    python scripts/train_graph_vae.py --max-samples 1000 --epochs 5

    # ZINC250K training
    python scripts/train_graph_vae.py --smiles-file data/raw/zinc250k.csv --epochs 50

    # ZINC 1M training
    python scripts/train_graph_vae.py --smiles-file data/raw/zinc_1m.csv --epochs 50

    # Full ZINC 22.8M (auto-streams, no RAM overflow)
    python scripts/train_graph_vae.py --smiles-file data/raw/zinc_full.csv --epochs 50

    # Custom config
    python scripts/train_graph_vae.py \\
        --smiles-file data/raw/zinc_1m.csv \\
        --max-nodes 38 \\
        --latent-dim 128 \\
        --hidden-dim 256 \\
        --batch-size 128 \\
        --lr 0.001 \\
        --epochs 50 \\
        --kl-weight 0.005
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from app.services.graph_molecule_service import (
    load_graph_dataset,
    create_graph_dataloader,
    initialize_graph_vae,
    train_graph_vae,
    evaluate_graph_generation,
    GRAPH_VAE_DEFAULTS,
)
from app.models.training_utils import get_training_device


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: GraphVAE Training for Molecular Generation"
    )

    # -----------------------------------------------------------------------
    # CLI Arguments
    # -----------------------------------------------------------------------

    # Data
    parser.add_argument(
        "--smiles-file", type=str, default=None,
        help="Path to SMILES file (.smi or .csv)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of molecules (for quick testing)"
    )

    # Model architecture
    parser.add_argument(
        "--max-nodes", type=int,
        default=GRAPH_VAE_DEFAULTS["max_nodes"],
        help="Maximum atoms per molecule"
    )
    parser.add_argument(
        "--latent-dim", type=int,
        default=GRAPH_VAE_DEFAULTS["latent_dim"],
        help="Latent space dimension"
    )
    parser.add_argument(
        "--hidden-dim", type=int,
        default=GRAPH_VAE_DEFAULTS["hidden_dim"],
        help="Hidden layer dimension"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size", type=int,
        default=GRAPH_VAE_DEFAULTS["batch_size"],
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float,
        default=GRAPH_VAE_DEFAULTS["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int,
        default=GRAPH_VAE_DEFAULTS["num_epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--kl-weight", type=float,
        default=GRAPH_VAE_DEFAULTS["kl_weight"],
        help="Fixed KL divergence weight (used when cyclical annealing is disabled)"
    )

    # Cyclical β annealing (TGVAE, Paper 4)
    parser.add_argument(
        "--beta-max", type=float,
        default=GRAPH_VAE_DEFAULTS["beta_max"],
        help="Maximum β value during cyclical annealing (default: 0.05)"
    )
    parser.add_argument(
        "--num-cycles", type=int,
        default=GRAPH_VAE_DEFAULTS["num_cycles"],
        help="Number of cyclical annealing cycles (default: 4)"
    )
    parser.add_argument(
        "--annealing-ratio", type=float,
        default=GRAPH_VAE_DEFAULTS["annealing_ratio"],
        help="Fraction of each cycle spent ramping up β (default: 0.5)"
    )
    parser.add_argument(
        "--no-cyclical-annealing", action="store_true",
        help="Disable cyclical annealing, use fixed --kl-weight instead"
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N epochs"
    )

    # Evaluation
    parser.add_argument(
        "--eval-samples", type=int, default=100,
        help="Number of molecules to generate for evaluation"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip post-training evaluation"
    )

    args = parser.parse_args()

    # ===================================================================
    # Banner — System & Hyperparameter Overview
    # ===================================================================
    print("="  * 65)
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Pharma-Z  ·  Phase 2: RGCVAE-Inspired GraphVAE   ║")
    print("  ║  RGIN + Cyclicalβ + ValenceDecoding + Rich28d     ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print("=" * 65)

    # Device detection
    device = get_training_device()
    print(f"\n  Device:        {device}")
    if device.type == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM:          {vram:.1f} GB")

    use_cyclical = not args.no_cyclical_annealing
    print(f"\n  Max nodes:     {args.max_nodes}")
    print(f"  Latent dim:    {args.latent_dim}")
    print(f"  Hidden dim:    {args.hidden_dim}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  LR:            {args.lr}")
    print(f"  Epochs:        {args.epochs}")
    if use_cyclical:
        print(f"  KL schedule:   Cyclical (β_max={args.beta_max}, {args.num_cycles} cycles)")
    else:
        print(f"  KL schedule:   Fixed (β={args.kl_weight})")
    print(f"  Encoder:       RGIN (3-layer, bond-aware)")
    print(f"  Node features: 28d (type+val+chg+chiral+hybrid)")
    print(f"  Pooling:       global_add_pool")
    print(f"  Node loss:     Cross-entropy")

    if args.max_samples:
        print(f"\n  Max samples:   {args.max_samples}")

    # ===================================================================
    # Step 1: Load Dataset (auto-detects streaming for large files)
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 1/4] Loading molecular graph dataset")
    print(f"{'=' * 65}")

    t0 = time.time()
    dataset = load_graph_dataset(
        smiles_file=args.smiles_file,
        max_nodes=args.max_nodes,
        max_samples=args.max_samples,
    )
    load_time = time.time() - t0
    print(f"  ✓ Dataset: {len(dataset):,} molecular graphs ({load_time:.1f}s)")

    # ===================================================================
    # Step 2: Create DataLoader
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 2/4] Creating DataLoader")
    print(f"{'=' * 65}")

    t1 = time.time()
    loader = create_graph_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Must be 0 — nohup/background processes deadlock with workers
    )
    print(f"  ✓ DataLoader ready ({time.time()-t1:.1f}s)")

    # ===================================================================
    # Step 3: Initialize Model & Train
    # ===================================================================
    print(f"\n{'=' * 65}")
    print("[Step 3/4] Training Enhanced GraphVAE")
    print(f"{'=' * 65}\n")

    # Initialize model
    model = initialize_graph_vae(
        max_nodes=args.max_nodes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Train
    t_train = time.time()
    history = train_graph_vae(
        data_loader=loader,
        model=model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        save_every=args.save_every,
        # Cyclical β annealing (TGVAE)
        use_cyclical_annealing=use_cyclical,
        beta_max=args.beta_max,
        num_cycles=args.num_cycles,
        annealing_ratio=args.annealing_ratio,
    )
    train_time = time.time() - t_train

    # Training summary
    print(f"\n  ✓ Training complete in {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"  ✓ Final loss: {history['total_loss'][-1]:.4f}")
    print(f"    Adj:  {history['recon_adj_loss'][-1]:.4f}")
    print(f"    Node: {history['recon_node_loss'][-1]:.4f}")
    print(f"    Bond: {history['recon_bond_loss'][-1]:.4f}")
    print(f"    KL:   {history['kl_loss'][-1]:.4f}")

    # ===================================================================
    # Step 4: Evaluate — Validity, Uniqueness, Novelty
    # ===================================================================
    if not args.skip_eval:
        print(f"\n{'=' * 65}")
        print("[Step 4/4] Evaluating generation quality")
        print(f"{'=' * 65}")

        print(f"\n  Generating {args.eval_samples} molecules...")
        metrics = evaluate_graph_generation(
            num_samples=args.eval_samples,
            model=model,
            temperature=1.0,
        )

        # Pretty-print metrics table
        v = metrics['validity']
        u = metrics['uniqueness']
        n = metrics['novelty']
        print(f"\n┌───────────────────────────────────────┐")
        print(f"  │  GraphVAE Evaluation Metrics          │")
        print(f"  ├───────────────────────────────────────┤")
        print(f"  │  Generated:  {metrics['num_generated']:>6}                  │")
        print(f"  │  Valid:      {metrics['num_valid']:>6}  ({v:.1%})            │")
        print(f"  │  Unique:     {metrics['num_unique']:>6}  ({u:.1%})            │")
        print(f"  │  Novelty:            {n:.1%}              │")
        print(f"  └───────────────────────────────────────┘")

        # Show sample generated molecules
        if metrics.get("sample_molecules"):
            print(f"\n  Sample valid molecules:")
            for i, smi in enumerate(metrics["sample_molecules"][:8]):
                print(f"    {i+1}. {smi}")

        # Save metrics to JSON
        metrics_path = os.path.join(
            "data", "saved_models", "graph_vae", "eval_metrics.json"
        )
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓ Metrics saved: {metrics_path}")

    # ===================================================================
    # Done — Summary
    # ===================================================================
    print(f"\n{'=' * 65}")
    print(f"  ✓ Phase 2 Complete — RGCVAE-Inspired GraphVAE Trained!")
    print(f"{'=' * 65}")
    print(f"\n  Checkpoints: data/saved_models/graph_vae/")
    print(f"  Metrics:     data/saved_models/graph_vae/eval_metrics.json")
    print(f"\n  Enhancements applied:")
    print(f"    ✓ RGIN encoder (per-bond-type MLPs, WL-equivalent)")
    print(f"    ✓ Rich 28d atom features (valence+charge+chirality+hybrid)")
    print(f"    ✓ Cyclical β annealing ({args.num_cycles} cycles, β_max={args.beta_max})" if use_cyclical else f"    ✓ Fixed β = {args.kl_weight}")
    print(f"    ✓ Cross-entropy node loss")
    print(f"    ✓ global_add_pool")
    print(f"    ✓ Valence-masked generation")
    print(f"\n  To generate molecules via API:")
    print(f"    uvicorn app.main:app --reload")
    print(f"    POST /api/v1/molecules/generate-graph")
    print(f"\n  Next: Phase 3 — Binding Affinity Prediction")
    print(f"    python scripts/train_deep_dta.py")


if __name__ == "__main__":
    main()
