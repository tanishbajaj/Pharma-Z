"""
Graph Molecule Service — Training, Generation & Evaluation for RGCVAE-Inspired GraphVAE

Business logic layer for the Stage 2 molecule generator.
Handles dataset loading, model training, molecule generation,
and evaluation metrics.

Enhanced with:
    - Cyclical β annealing (TGVAE, Nguyen & Karolak 2024)
    - Valence-masked molecule generation (RGCVAE + JT-VAE)
    - RGIN encoder support (edge_type forwarding)
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
try:
    from torch_geometric.loader import DataLoader  # type: ignore[import-untyped]
except ImportError:
    DataLoader = None  # torch-geometric not installed; GraphVAE training unavailable

from app.config import settings
from app.models.graph_vae import (
    GraphVAE,
    compute_graph_vae_loss,
    checkpoint_has_node_existence_head,
    NODE_EXISTENCE_KEY_PREFIX,
)
from app.models.graph_dataset import ZINCGraphDataset
from app.models.graph_streaming_dataset import StreamingGraphDataset
from app.models.graph_utils import (
    adjacency_to_smiles,
    adjacency_to_smiles_valence_masked,
    NODE_FEATURE_DIM,
    ATOM_TYPES,
)
from app.models.training_utils import (
    get_training_device,
    save_model_checkpoint,
    load_model_checkpoint,
)
from app.utils.logging_config import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRAPH_VAE_DEFAULTS = {
    "max_nodes": 25,          # V2: reduced from 38 → 25 (covers 95%+ of ZINC)
    "latent_dim": 128,
    "hidden_dim": 256,
    "batch_size": 64,         # V2: reduced from 128 → 64 for stability
    "learning_rate": 3e-4,
    "num_epochs": 100,        # V2: increased from 50 → 100 (adj loss still improving)
    # Cyclical β annealing (Paper 4: TGVAE)
    "kl_weight": 0.005,       # legacy fixed weight (used if annealing disabled)
    "beta_max": 0.05,         # V2: increased from 0.01 → 0.05 for better regularization
    "num_cycles": 4,          # number of annealing cycles
    "annealing_ratio": 0.5,   # fraction of each cycle spent ramping up
    "use_cyclical_annealing": True,
    # Weighted reconstruction loss
    "lambda_adj": 1.0,
    "lambda_node": 1.0,
    "lambda_bond": 1.0,
    "lambda_exist": 2.0,      # V2: strong weight for node existence learning
}


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_graph_vae_model: Optional[GraphVAE] = None
_device: Optional[torch.device] = None


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

STREAMING_THRESHOLD = 500_000  # Auto-stream if dataset > 500K molecules


def load_graph_dataset(
    smiles_file: Optional[str] = None,
    max_nodes: int = None,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force_streaming: bool = False,
    shuffle_buffer: int = 50_000,
):
    """
    Load the ZINC graph dataset.

    Automatically switches to streaming mode for large datasets (>500K)
    to avoid RAM overflow.

    Args:
        smiles_file:     Path to SMILES file (.smi or .csv).
        max_nodes:       Maximum atoms per molecule.
        max_samples:     Limit samples (for testing).
        cache_dir:       Directory for processed cache.
        force_streaming: Force streaming mode regardless of file size.

    Returns:
        ZINCGraphDataset or StreamingGraphDataset ready for DataLoader.
    """
    if max_nodes is None:
        max_nodes = GRAPH_VAE_DEFAULTS["max_nodes"]

    if smiles_file is None:
        # Try common locations
        candidates = [
            os.path.join(settings.data_raw_dir, "zinc_merged.smi"),
            os.path.join(settings.data_raw_dir, "zinc_full.csv"),
            os.path.join(settings.data_raw_dir, "zinc250k.csv"),
            os.path.join(settings.data_raw_dir, "zinc_1m.smi"),
            os.path.join(settings.data_raw_dir, "zinc_1m.csv"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                smiles_file = candidate
                break

        if smiles_file is None:
            raise FileNotFoundError(
                f"No SMILES file found. Tried: {candidates}\n"
                f"Run 'python scripts/download_zinc22.py' or place a .smi/.csv in {settings.data_raw_dir}"
            )

    # Check file size to decide mode
    use_streaming = force_streaming
    if not use_streaming:
        line_count = _count_file_lines(smiles_file)
        effective = min(line_count, max_samples or float("inf"))
        use_streaming = effective > STREAMING_THRESHOLD
        if use_streaming:
            logger.info(
                f"Large dataset detected ({line_count:,} lines) "
                f"-> STREAMING mode (low RAM)"
            )

    if use_streaming:
        dataset = StreamingGraphDataset(
            smiles_file=smiles_file,
            max_nodes=max_nodes,
            chunk_size=10000,
            max_samples=max_samples,
            shuffle_buffer=shuffle_buffer,
        )
        logger.info(
            f"Streaming dataset: ~{len(dataset):,} molecules | "
            f"max_nodes={max_nodes}"
        )
        return dataset

    # In-memory mode for smaller datasets
    if cache_dir is None:
        cache_dir = os.path.join(settings.data_processed_dir, "graph_cache")

    logger.info(f"Loading graph dataset from: {smiles_file}")

    dataset = ZINCGraphDataset(
        root=cache_dir,
        smiles_file=smiles_file,
        max_nodes=max_nodes,
        max_samples=max_samples,
    )

    logger.info(f"Dataset loaded: {len(dataset):,} molecular graphs")
    return dataset


def _count_file_lines(filepath: str) -> int:
    """Fast line count for streaming decision."""
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return max(count - 1, 0)  # subtract header


def create_graph_dataloader(
    dataset,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyG DataLoader from the dataset.

    Handles both InMemoryDataset and IterableDataset (streaming).
    """
    if batch_size is None:
        batch_size = GRAPH_VAE_DEFAULTS["batch_size"]

    # DataLoader cannot shuffle an IterableDataset — StreamingGraphDataset
    # shuffles internally with a reservoir buffer instead.
    is_streaming = isinstance(dataset, StreamingGraphDataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and not is_streaming),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    mode = "STREAMING" if is_streaming else "IN-MEMORY"
    approx_batches = len(dataset) // batch_size if hasattr(dataset, '__len__') else '?'
    detail = ""
    if is_streaming:
        detail = (f", workers={num_workers}, "
                  f"shuffle_buffer={dataset.shuffle_buffer:,}")
    logger.info(
        f"DataLoader [{mode}]: ~{len(dataset):,} graphs, "
        f"batch_size={batch_size}, "
        f"~{approx_batches} batches/epoch{detail}"
    )
    return loader


# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------

def initialize_graph_vae(
    max_nodes: int = None,
    latent_dim: int = None,
    hidden_dim: int = None,
) -> GraphVAE:
    """Create and return a new GraphVAE model."""
    global _graph_vae_model, _device

    if max_nodes is None:
        max_nodes = GRAPH_VAE_DEFAULTS["max_nodes"]
    if latent_dim is None:
        latent_dim = GRAPH_VAE_DEFAULTS["latent_dim"]
    if hidden_dim is None:
        hidden_dim = GRAPH_VAE_DEFAULTS["hidden_dim"]

    _device = get_training_device()

    _graph_vae_model = GraphVAE(
        max_nodes=max_nodes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        node_feature_dim=NODE_FEATURE_DIM,
    ).to(_device)

    total_params = sum(p.numel() for p in _graph_vae_model.parameters())
    trainable_params = sum(
        p.numel() for p in _graph_vae_model.parameters() if p.requires_grad
    )

    logger.info(
        f"GraphVAE initialized: "
        f"{total_params:,} total params, "
        f"{trainable_params:,} trainable, "
        f"max_nodes={max_nodes}, latent_dim={latent_dim}, "
        f"hidden_dim={hidden_dim}, device={_device}"
    )

    return _graph_vae_model


def load_graph_vae_checkpoint(checkpoint_path: str) -> GraphVAE:
    """
    Build a GraphVAE matching `checkpoint_path` and load its weights.

    Dimensions are read from the checkpoint rather than GRAPH_VAE_DEFAULTS,
    which drift between training runs (V1 trained at max_nodes=38, the V2
    default is 25).

    V1 checkpoints have no learned node-existence head. Those are loaded with
    the head left uninitialized and generation switched back to the V1
    max-edge heuristic, so they reproduce their original output instead of
    reading a random head — which would look like molecules and be noise.

    Raises:
        RuntimeError: if the checkpoint mismatches in any other way.
    """
    global _graph_vae_model

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = raw.get("model_state_dict", raw)

    max_nodes = GRAPH_VAE_DEFAULTS["max_nodes"]
    latent_dim = GRAPH_VAE_DEFAULTS["latent_dim"]
    hidden_dim = GRAPH_VAE_DEFAULTS["hidden_dim"]

    # max_nodes, hidden_dim → decoder.pos_embed.weight; latent_dim → encoder.fc_mu
    if "decoder.pos_embed.weight" in state_dict:
        max_nodes, hidden_dim = state_dict["decoder.pos_embed.weight"].shape
    if "encoder.fc_mu.bias" in state_dict:
        latent_dim = state_dict["encoder.fc_mu.bias"].shape[0]

    logger.info(
        f"Detected checkpoint architecture: max_nodes={max_nodes}, "
        f"latent_dim={latent_dim}, hidden_dim={hidden_dim}"
    )

    model = initialize_graph_vae(
        max_nodes=int(max_nodes),
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
    )

    has_head = checkpoint_has_node_existence_head(state_dict)
    load_model_checkpoint(
        model,
        checkpoint_path,
        allow_missing=() if has_head else (NODE_EXISTENCE_KEY_PREFIX,),
    )
    model.use_learned_node_existence = has_head

    if not has_head:
        logger.warning(
            f"'{checkpoint_path}' is a V1 checkpoint (no learned node-existence "
            f"head). Falling back to the V1 max-edge heuristic for node "
            f"existence, which is the setup that produced disconnected "
            f"'C.C.C...' fragments. Retrain to use the learned head."
        )

    _graph_vae_model = model
    return model


# ---------------------------------------------------------------------------
# Cyclical β Annealing (TGVAE, Paper 4)
# ---------------------------------------------------------------------------

def cyclical_beta_schedule(
    epoch: int,
    total_epochs: int,
    num_cycles: int = 4,
    beta_max: float = 0.05,
    ratio: float = 0.5,
) -> float:
    """
    Compute cyclical β (KL weight) for a given epoch.

    From TGVAE (Nguyen & Karolak, 2024): cyclical annealing schedules
    outperform monotonic schedules on MOSES metrics. Each cycle ramps
    β from 0 to beta_max, then holds, allowing the model to periodically
    explore unconstrained reconstruction before being regularized.

    Args:
        epoch:        Current epoch (1-indexed).
        total_epochs: Total number of training epochs.
        num_cycles:   Number of annealing cycles (default 4).
        beta_max:     Maximum β value (default 0.05).
        ratio:        Fraction of each cycle spent ramping up (default 0.5).

    Returns:
        β value for the current epoch.
    """
    cycle_length = total_epochs / num_cycles
    position = ((epoch - 1) % cycle_length) / cycle_length
    if position < ratio:
        return beta_max * (position / ratio)
    return beta_max


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_graph_vae(
    data_loader: DataLoader,
    model: Optional[GraphVAE] = None,
    num_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    kl_weight: Optional[float] = None,
    lambda_adj: Optional[float] = None,
    lambda_node: Optional[float] = None,
    lambda_bond: Optional[float] = None,
    save_every: int = 10,
    checkpoint_dir: Optional[str] = None,
    # Cyclical β annealing params (Paper 4: TGVAE)
    use_cyclical_annealing: Optional[bool] = None,
    beta_max: Optional[float] = None,
    num_cycles: Optional[int] = None,
    annealing_ratio: Optional[float] = None,
    start_epoch: int = 0,  # Epoch offset for resuming: keeps β schedule correct
) -> Dict:
    """
    Train the GraphVAE model with weighted reconstruction loss (Paper Eq. 3).

    Args:
        data_loader:    PyG DataLoader with molecular graph batches.
        model:          GraphVAE model (creates new if None).
        num_epochs:     Number of training epochs.
        learning_rate:  Adam learning rate.
        kl_weight:      Weight for KL divergence term (β).
        lambda_adj:     Weight for adjacency reconstruction (λ_A).
        lambda_node:    Weight for node feature reconstruction (λ_F).
        lambda_bond:    Weight for bond type reconstruction (λ_E).
        save_every:     Save checkpoint every N epochs.
        checkpoint_dir: Directory for model checkpoints.

    Returns:
        Training history dict.
    """
    if model is None:
        model = initialize_graph_vae()

    if num_epochs is None:
        num_epochs = GRAPH_VAE_DEFAULTS["num_epochs"]
    if learning_rate is None:
        learning_rate = GRAPH_VAE_DEFAULTS["learning_rate"]
    if kl_weight is None:
        kl_weight = GRAPH_VAE_DEFAULTS["kl_weight"]
    if lambda_adj is None:
        lambda_adj = GRAPH_VAE_DEFAULTS["lambda_adj"]
    if lambda_node is None:
        lambda_node = GRAPH_VAE_DEFAULTS["lambda_node"]
    if lambda_bond is None:
        lambda_bond = GRAPH_VAE_DEFAULTS["lambda_bond"]
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(settings.saved_models_dir, "graph_vae")

    # Cyclical β annealing defaults (Paper 4: TGVAE)
    if use_cyclical_annealing is None:
        use_cyclical_annealing = GRAPH_VAE_DEFAULTS["use_cyclical_annealing"]
    if beta_max is None:
        beta_max = GRAPH_VAE_DEFAULTS["beta_max"]
    if num_cycles is None:
        num_cycles = GRAPH_VAE_DEFAULTS["num_cycles"]
    if annealing_ratio is None:
        annealing_ratio = GRAPH_VAE_DEFAULTS["annealing_ratio"]

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler: reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    history = {
        "total_loss": [],
        "recon_adj_loss": [],
        "recon_node_loss": [],
        "recon_bond_loss": [],
        "node_exist_loss": [],
        "kl_loss": [],
        "kl_weight": [],  # track β over time
        "epoch_time": [],
    }

    annealing_str = (
        f"cyclical (β_max={beta_max}, {num_cycles} cycles, ratio={annealing_ratio})"
        if use_cyclical_annealing
        else f"fixed (β={kl_weight})"
    )
    logger.info(
        f"Starting RGCVAE training: "
        f"epochs={num_epochs}, lr={learning_rate}, "
        f"KL schedule: {annealing_str}, "
        f"λ_A={lambda_adj}, λ_F={lambda_node}, λ_E={lambda_bond}, "
        f"device={device}"
    )

    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Compute β for this epoch, offset by start_epoch when resuming
        if use_cyclical_annealing:
            current_kl_weight = cyclical_beta_schedule(
                epoch + start_epoch, num_epochs + start_epoch,
                num_cycles, beta_max, annealing_ratio
            )
        else:
            current_kl_weight = kl_weight

        epoch_losses = {
            "total": 0.0, "adj": 0.0, "exist": 0.0,
            "node": 0.0, "bond": 0.0, "kl": 0.0
        }
        num_batches = 0

        for batch in data_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            # Forward pass — returns 6-tuple (adj_pred, node_pred, bond_pred, mu, logvar, adj_logits)
            adj_pred, node_pred, bond_pred, mu, logvar, adj_logits, node_exist_logits = model(batch)

            # Get ground truth tensors
            adj_true = batch.adj.view(-1, model.max_nodes, model.max_nodes)
            node_true = batch.padded_x.view(
                -1, model.max_nodes, NODE_FEATURE_DIM
            )
            bond_true = batch.bond_types.view(
                -1, model.max_nodes, model.max_nodes
            )

            # Compute weighted loss (V2: includes node existence head)
            losses = compute_graph_vae_loss(
                adj_pred=adj_pred,
                node_pred=node_pred,
                bond_pred=bond_pred,
                adj_true=adj_true,
                node_true=node_true,
                bond_true=bond_true,
                mu=mu,
                logvar=logvar,
                adj_logits=adj_logits,
                node_exist_logits=node_exist_logits,
                kl_weight=current_kl_weight,
                lambda_adj=lambda_adj,
                lambda_node=lambda_node,
                lambda_bond=lambda_bond,
            )

            # Guard: skip batch if loss is NaN or Inf
            if not torch.isfinite(losses["total_loss"]):
                logger.warning(
                    f"  ⚠ Skipping batch (NaN/Inf loss detected)"
                )
                optimizer.zero_grad()  # clear any partial gradients
                continue

            # Backward
            losses["total_loss"].backward()

            # Gradient clipping for stability (1.0 is aggressive but safe)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_losses["total"] += losses["total_loss"].item()
            epoch_losses["adj"] += losses["recon_adj_loss"].item()
            epoch_losses["exist"] += losses["node_exist_loss"].item()
            epoch_losses["node"] += losses["recon_node_loss"].item()
            epoch_losses["bond"] += losses["recon_bond_loss"].item()
            epoch_losses["kl"] += losses["kl_loss"].item()
            num_batches += 1

        # Average losses
        avg = {k: v / num_batches for k, v in epoch_losses.items()}
        epoch_time = time.time() - epoch_start

        history["total_loss"].append(avg["total"])
        history["recon_adj_loss"].append(avg["adj"])
        history["node_exist_loss"].append(avg.get("exist", 0.0))
        history["recon_node_loss"].append(avg["node"])
        history["recon_bond_loss"].append(avg["bond"])
        history["kl_loss"].append(avg["kl"])
        history["kl_weight"].append(current_kl_weight)
        history["epoch_time"].append(epoch_time)

        # LR scheduler step
        scheduler.step(avg["total"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging (V2: includes node existence loss)
        logger.info(
            f"Epoch {epoch:>3}/{num_epochs} | "
            f"Total: {avg['total']:.4f} | "
            f"Adj: {avg['adj']:.4f} | "
            f"Exist: {avg.get('exist', 0):.4f} | "
            f"Node: {avg['node']:.4f} | "
            f"Bond: {avg['bond']:.4f} | "
            f"KL: {avg['kl']:.4f} | "
            f"β: {current_kl_weight:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint — use absolute epoch number (relative + start_epoch offset)
        if epoch % save_every == 0 or epoch == num_epochs:
            abs_epoch = epoch + start_epoch
            ckpt_path = os.path.join(
                checkpoint_dir, f"graph_vae_epoch_{abs_epoch}.pt"
            )
            save_model_checkpoint(
                model, optimizer, abs_epoch, avg["total"], ckpt_path
            )

    logger.info("RGCVAE training complete.")
    return history


# ---------------------------------------------------------------------------
# Molecule Generation
# ---------------------------------------------------------------------------

def generate_molecules_graph(
    num_molecules: int = 10,
    model: Optional[GraphVAE] = None,
    checkpoint_path: Optional[str] = None,
    temperature: float = 1.0,
    adj_threshold: float = 0.5,
) -> List[Dict]:
    """
    Generate novel molecules by sampling from the GraphVAE latent space.

    Args:
        num_molecules:   Number of molecules to generate.
        model:           Trained GraphVAE (loads from checkpoint if None).
        checkpoint_path: Path to model checkpoint.
        temperature:     Sampling temperature.
        adj_threshold:   Threshold for adjacency matrix binarization.

    Returns:
        List of dicts with: smiles, is_valid, adj_matrix, node_types.
    """
    global _graph_vae_model

    if model is None:
        if _graph_vae_model is None:
            if checkpoint_path:
                _graph_vae_model = load_graph_vae_checkpoint(checkpoint_path)
            else:
                raise ValueError(
                    "No model available. Train first or provide checkpoint."
                )
        model = _graph_vae_model

    device = next(model.parameters()).device

    # Generate — now returns 3-tuple with bond_pred
    adj_pred, node_pred, bond_pred = model.generate(
        num_molecules=num_molecules,
        device=device,
        temperature=temperature,
    )

    # Convert to molecules using VALENCE-MASKED generation (RGCVAE + JT-VAE)
    results = []
    adj_np = adj_pred.cpu().numpy()
    node_np = node_pred.cpu().numpy()
    bond_np = bond_pred.cpu().numpy()

    for i in range(num_molecules):
        adj_matrix = adj_np[i]
        node_features = node_np[i]
        bond_logits = bond_np[i]  # [max_nodes, max_nodes, NUM_BOND_TYPES]

        # PRIMARY: Use valence-masked generation (RGCVAE approach)
        # This iteratively places edges while respecting valence budgets
        smiles = adjacency_to_smiles_valence_masked(
            adj_matrix,
            node_features=node_features,
            bond_logits=bond_logits,
            threshold=adj_threshold,
        )

        # FALLBACK: If valence-masked fails, try threshold + MST approach
        if smiles is None:
            num_atom_types = len(ATOM_TYPES) + 1
            atom_type_logits = node_features[:, :num_atom_types]
            node_types = np.argmax(atom_type_logits, axis=1)
            bond_types = np.argmax(bond_logits, axis=2)

            smiles = adjacency_to_smiles(
                adj_matrix,
                node_types=node_types,
                bond_types=bond_types,
                threshold=adj_threshold,
                enforce_connected=True,
            )

        results.append({
            "smiles": smiles,
            "is_valid": smiles is not None,
            "properties": {},
        })

        # Compute properties for valid molecules
        if smiles is not None:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, QED as QEDModule
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    results[-1]["properties"] = {
                        "molecular_weight": round(
                            float(Descriptors.MolWt(mol)), 2
                        ),
                        "logp": round(
                            float(Descriptors.MolLogP(mol)), 2
                        ),
                        "qed_score": round(
                            float(QEDModule.qed(mol)), 4
                        ),
                        "num_atoms": mol.GetNumAtoms(),
                        "num_rings": Descriptors.RingCount(mol),
                    }
            except Exception:
                pass

    valid_count = sum(1 for r in results if r["is_valid"])
    logger.info(
        f"Generated {num_molecules} molecules: "
        f"{valid_count} valid ({100 * valid_count / num_molecules:.1f}%)"
    )

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_graph_generation(
    num_samples: int = 100,
    model: Optional[GraphVAE] = None,
    temperature: float = 1.0,
    training_smiles: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate the quality of generated molecules.

    Metrics:
        - Validity:   fraction that parse as valid SMILES
        - Uniqueness: fraction of valid that are unique
        - Novelty:    fraction not in training set
    """
    results = generate_molecules_graph(
        num_molecules=num_samples,
        model=model,
        temperature=temperature,
    )

    valid_smiles = [r["smiles"] for r in results if r["is_valid"]]
    unique_smiles = set(valid_smiles)

    validity = len(valid_smiles) / num_samples if num_samples > 0 else 0
    uniqueness = (
        len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
    )

    novelty = 0.0
    if training_smiles is not None and unique_smiles:
        training_set = set(training_smiles)
        novel = unique_smiles - training_set
        novelty = len(novel) / len(unique_smiles)

    metrics = {
        "num_generated": num_samples,
        "num_valid": len(valid_smiles),
        "num_unique": len(unique_smiles),
        "validity": round(validity, 4),
        "uniqueness": round(uniqueness, 4),
        "novelty": round(novelty, 4),
        "sample_molecules": valid_smiles[:10],
    }

    logger.info(
        f"Evaluation: validity={validity:.2%}, "
        f"uniqueness={uniqueness:.2%}, "
        f"novelty={novelty:.2%}"
    )

    return metrics
