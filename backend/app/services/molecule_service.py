"""
Molecule Service

Business logic for molecule generation using the SMILES VAE.
Handles data loading, training, generation, and validation.
"""

import os
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, AllChem, BRICS
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import DataStructs

# Silence RDKit SMILES parse errors and deprecation warnings
# These are expected when validating generated (sometimes invalid) SMILES
RDLogger.DisableLog('rdApp.*')

from app.config import settings
from app.models.smiles_tokenizer import SmilesTokenizer
from app.models.smiles_vae import SmilesVAE, compute_vae_loss
from app.models.training_utils import (
    get_training_device,
    get_mixed_precision_context,
    get_gradient_scaler,
    save_model_checkpoint,
    load_model_checkpoint,
)
from app.utils.logging_config import logger
from app.utils.constants import (
    MAX_MOLECULE_LENGTH,
    DEFAULT_LATENT_DIM,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    ZINC250K_FILENAME,
)


# ---------------------------------------------------------------------------
# Global instances (lazily initialized)
# ---------------------------------------------------------------------------

_tokenizer: Optional[SmilesTokenizer] = None
_vae_model: Optional[SmilesVAE] = None
_device: Optional[torch.device] = None


def get_tokenizer() -> SmilesTokenizer:
    """Get or create the shared tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = SmilesTokenizer(max_length=MAX_MOLECULE_LENGTH)
    return _tokenizer


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_zinc_dataset(
    file_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    smiles_column: str = "smiles",
) -> List[str]:
    """
    Load SMILES strings from the ZINC250K CSV file.

    Args:
        file_path:     Path to the CSV file. Defaults to data/raw/zinc250k.csv.
        max_samples:   Limit number of samples (for quick testing).
        smiles_column: Name of the column containing SMILES strings.

    Returns:
        List of SMILES strings.
    """
    if file_path is None:
        file_path = os.path.join(settings.data_raw_dir, ZINC250K_FILENAME)

    logger.info(f"Loading ZINC dataset from: {file_path}")

    dataframe = pd.read_csv(file_path)

    if smiles_column not in dataframe.columns:
        # Try common alternative column names
        possible_columns = ["smiles", "SMILES", "Smiles", "mol", "molecule"]
        found_column: str = str(dataframe.columns[0])  # fallback default
        for col in possible_columns:
            if col in dataframe.columns:
                found_column = col
                break
        smiles_column = found_column

    smiles_list: List[str] = [
        s for s in (str(x) for x in dataframe[smiles_column].dropna().tolist())
        if s and s != "nan" and s.strip()  # drop empty/nan strings
    ]

    # Filter by length — min 5 chars, max MAX_MOLECULE_LENGTH
    smiles_list = [s for s in smiles_list if 5 <= len(s) <= MAX_MOLECULE_LENGTH]

    if max_samples is not None:
        smiles_list = smiles_list[:max_samples]

    logger.info(f"Loaded {len(smiles_list)} SMILES strings")
    return smiles_list


def prepare_training_data(
    smiles_list: List[str],
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Tokenize SMILES strings and create a PyTorch DataLoader.

    Args:
        smiles_list: List of SMILES strings.
        batch_size:  Batch size for training.
        shuffle:     Whether to shuffle the data.
        num_workers: Number of parallel data loading workers.

    Returns:
        DataLoader yielding (input_tokens, target_tokens) batches.
    """
    if batch_size is None:
        batch_size = settings.default_batch_size

    tokenizer = get_tokenizer()
    encoded_sequences = tokenizer.encode_batch(smiles_list)

    # Input = full sequence, Target = same sequence (autoencoder reconstruction)
    input_tensor = torch.tensor(encoded_sequences, dtype=torch.long)
    target_tensor = torch.tensor(encoded_sequences, dtype=torch.long)

    dataset = TensorDataset(input_tensor, target_tensor)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 0 avoids multiprocessing deadlocks in background/nohup
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"Prepared DataLoader: {len(dataset)} samples, "
        f"batch_size={batch_size}, "
        f"num_batches={len(data_loader)}"
    )

    return data_loader


# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------

def initialize_vae_model(
    latent_dim: int = DEFAULT_LATENT_DIM,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
) -> SmilesVAE:
    """Create and return a new SMILES VAE model on the best available device."""
    global _vae_model, _device

    _device = get_training_device()
    tokenizer = get_tokenizer()

    _vae_model = SmilesVAE(
        vocabulary_size=tokenizer.vocabulary_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_length=MAX_MOLECULE_LENGTH + 2,
    ).to(_device)

    total_parameters = sum(p.numel() for p in _vae_model.parameters())
    trainable_parameters = sum(p.numel() for p in _vae_model.parameters() if p.requires_grad)

    logger.info(
        f"SMILES VAE initialized: "
        f"{total_parameters:,} total params, "
        f"{trainable_parameters:,} trainable, "
        f"latent_dim={latent_dim}"
    )

    return _vae_model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_smiles_vae(
    data_loader: DataLoader,
    model: Optional[SmilesVAE] = None,
    num_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    kl_weight: float = 0.5,
    save_every: int = 10,
    checkpoint_dir: Optional[str] = None,
    kl_anneal_mode: str = "linear",
) -> dict:
    """
    Train the SMILES VAE model.

    Args:
        data_loader:    Training DataLoader.
        model:          VAE model (creates new if None).
        num_epochs:     Number of training epochs.
        learning_rate:  Optimizer learning rate.
        kl_weight:      Target weight for KL divergence loss.
        save_every:     Save checkpoint every N epochs.
        checkpoint_dir: Directory for saving checkpoints.
        kl_anneal_mode: KL annealing strategy:
                        'linear'  - ramp 0→kl_weight over warm-up epochs (default)
                        'cyclical' - cyclical annealing (Fu et al. 2019 / TGVAE)

    Returns:
        Dictionary with training history (losses per epoch).
    """
    if model is None:
        model = initialize_vae_model()

    if num_epochs is None:
        num_epochs = settings.default_epochs

    if learning_rate is None:
        learning_rate = settings.default_learning_rate

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(settings.saved_models_dir, "smiles_vae")

    device = next(model.parameters()).device
    tokenizer = get_tokenizer()

    # Lower LR (3e-4 per MOSES benchmark) for more stable training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,      # Full decay over all epochs
        eta_min=learning_rate * 0.01,  # Min LR = 1% of initial
    )
    scaler = get_gradient_scaler(device)

    training_history = {
        "total_loss": [],
        "reconstruction_loss": [],
        "kl_loss": [],
    }

    # KL annealing configuration
    kl_warm_up_epochs = min(10, num_epochs // 5)  # 10 epochs or 20% of total
    n_cycles = 4  # Number of cycles for cyclical annealing

    logger.info(
        f"Starting SMILES VAE training: "
        f"epochs={num_epochs}, lr={learning_rate}, "
        f"kl_weight={kl_weight} (anneal={kl_anneal_mode}), "
        f"device={device}"
    )

    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        for batch_input, batch_target in data_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()

            # Model forward in float16 (fast), loss in float32 (stable)
            # word_dropout_rate=0.15: mask 15% of decoder input tokens
            # Reduced from 0.25 → 0.15: less aggressive = better validity
            # Still enough to prevent posterior collapse (force z usage)
            with get_mixed_precision_context(device):
                reconstructed_logits, mu, log_var = model(
                    batch_input, batch_target, word_dropout_rate=0.15
                )

            # KL annealing schedule
            if kl_anneal_mode == "cyclical":
                # Cyclical annealing (Fu et al. 2019 / TGVAE paper)
                # Periodically ramp KL weight, allowing latent space to re-open
                cycle_length = num_epochs / n_cycles
                cycle_pos = (epoch - 1) % cycle_length
                ramp_fraction = cycle_pos / (cycle_length * 0.5)  # Ramp over first half
                annealed_kl_weight = kl_weight * min(1.0, ramp_fraction)
            else:
                # Linear annealing: weight ramps 0 → kl_weight over warm-up epochs
                annealed_kl_weight = kl_weight * min(1.0, epoch / kl_warm_up_epochs)

            # Cast outputs to float32 before loss computation
            losses = compute_vae_loss(
                reconstructed_logits=reconstructed_logits.float(),
                target_tokens=batch_target,
                mu=mu.float(),
                log_var=log_var.float(),
                kl_weight=annealed_kl_weight,
                pad_index=tokenizer.pad_index,
            )

            # Backward pass with gradient scaling + gradient clipping
            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_total_loss += losses["total_loss"].item()
            epoch_recon_loss += losses["reconstruction_loss"].item()
            epoch_kl_loss += losses["kl_loss"].item()
            num_batches += 1

            # Print progress every 500 batches
            if num_batches % 500 == 0:
                logger.info(
                    f"  Epoch {epoch} | Batch {num_batches} | "
                    f"Loss: {epoch_total_loss/num_batches:.4f}"
                )

        # Average losses for the epoch
        avg_total_loss = epoch_total_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches

        training_history["total_loss"].append(avg_total_loss)
        training_history["reconstruction_loss"].append(avg_recon_loss)
        training_history["kl_loss"].append(avg_kl_loss)

        current_lr = optimizer.param_groups[0]['lr']
        if kl_anneal_mode == "cyclical":
            cycle_length = num_epochs / n_cycles
            cycle_pos = (epoch - 1) % cycle_length
            ramp_fraction = cycle_pos / (cycle_length * 0.5)
            annealed_kl = kl_weight * min(1.0, ramp_fraction)
        else:
            annealed_kl = kl_weight * min(1.0, epoch / kl_warm_up_epochs)
        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Total: {avg_total_loss:.4f} | "
            f"Recon: {avg_recon_loss:.4f} | "
            f"KL: {avg_kl_loss:.4f} | "
            f"KL-w: {annealed_kl:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Step LR scheduler
        scheduler.step()

        # Save checkpoint periodically
        if epoch % save_every == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch}.pt")
            save_model_checkpoint(model, optimizer, epoch, avg_total_loss, checkpoint_path)

    logger.info("SMILES VAE training complete.")
    return training_history


# ---------------------------------------------------------------------------
# Molecule Generation
# ---------------------------------------------------------------------------

def generate_molecules(
    num_molecules: int = 10,
    model: Optional[SmilesVAE] = None,
    checkpoint_path: Optional[str] = None,
    temperature: float = 0.8,
) -> List[dict]:
    """
    Generate novel molecules by sampling from the VAE latent space.

    Args:
        num_molecules:   Number of molecules to generate.
        model:           Trained VAE model (loads from checkpoint if None).
        checkpoint_path: Path to the model checkpoint file.
        temperature:     Softmax temperature (0.8=focused/valid, 1.0=standard, 1.2=creative)
                         Default 0.8 gives best validity-diversity tradeoff.

    Returns:
        List of dicts with keys: smiles, is_valid, properties.
    """
    global _vae_model, _device

    tokenizer = get_tokenizer()

    if model is None:
        # Auto-discover latest checkpoint if none provided
        if _vae_model is None and checkpoint_path is None:
            checkpoint_dir = os.path.join(
                settings.saved_models_dir, "smiles_vae"
            )
            if os.path.isdir(checkpoint_dir):
                pt_files = [
                    f for f in os.listdir(checkpoint_dir)
                    if f.startswith("vae_epoch_") and f.endswith(".pt")
                ]
                if pt_files:
                    # Pick highest epoch: vae_epoch_100.pt > vae_epoch_50.pt
                    pt_files.sort(
                        key=lambda f: int(f.replace("vae_epoch_", "").replace(".pt", "")),
                        reverse=True,
                    )
                    checkpoint_path = os.path.join(checkpoint_dir, pt_files[0])
                    logger.info(f"Auto-discovered checkpoint: {checkpoint_path}")

        if _vae_model is None and checkpoint_path:
            # Auto-detect architecture from checkpoint
            import torch as _torch
            _ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            _sd = _ckpt.get("model_state_dict", _ckpt)

            # Detect dims:
            # hidden_dim  → encoder.lstm.weight_hh_l0 shape[1]
            # latent_dim  → encoder.fc_mu.bias shape[0]
            # embedding_dim → encoder.embedding.weight shape[1]
            _hidden_dim    = DEFAULT_HIDDEN_DIM
            _latent_dim    = DEFAULT_LATENT_DIM
            _embedding_dim = DEFAULT_EMBEDDING_DIM

            if "encoder.lstm.weight_hh_l0" in _sd:
                _hidden_dim = _sd["encoder.lstm.weight_hh_l0"].shape[1]
            if "encoder.fc_mu.bias" in _sd:
                _latent_dim = _sd["encoder.fc_mu.bias"].shape[0]
            if "encoder.embedding.weight" in _sd:
                _embedding_dim = _sd["encoder.embedding.weight"].shape[1]

            logger.info(
                f"Detected SMILES VAE architecture: "
                f"hidden_dim={_hidden_dim}, latent_dim={_latent_dim}, embedding_dim={_embedding_dim}"
            )
            _vae_model = initialize_vae_model(
                latent_dim=_latent_dim,
                embedding_dim=_embedding_dim,
                hidden_dim=_hidden_dim,
            )
            load_model_checkpoint(_vae_model, checkpoint_path)
        model = _vae_model

    if model is None:
        raise ValueError("No model available. Train a model first or provide a checkpoint path.")

    device = next(model.parameters()).device
    model.eval()

    generated_tokens = model.generate_molecules(
        num_molecules=num_molecules,
        start_token_index=tokenizer.start_index,
        device=device,
        temperature=temperature,
    )

    generated_smiles_list = tokenizer.decode_batch(generated_tokens.cpu().tolist())

    results = []
    for smiles_string in generated_smiles_list:
        molecule_info = {
            "smiles": smiles_string,
            "is_valid": False,
            "properties": {},
        }

        # Validate with RDKit
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is not None:
            molecule_info["is_valid"] = True
            molecule_info["properties"] = compute_molecule_properties(mol)

        results.append(molecule_info)

    valid_count = sum(1 for r in results if r["is_valid"])
    logger.info(
        f"Generated {num_molecules} molecules: "
        f"{valid_count} valid ({100 * valid_count / num_molecules:.1f}%)"
    )

    return results


# ---------------------------------------------------------------------------
# Validation & Properties
# ---------------------------------------------------------------------------

def validate_smiles(smiles_string: str) -> dict:
    """
    Validate a SMILES string and return its properties if valid.

    Returns:
        Dict with is_valid, canonical_smiles, and properties.
    """
    result = {
        "input_smiles": smiles_string,
        "is_valid": False,
        "canonical_smiles": None,
        "properties": {},
    }

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is not None:
        result["is_valid"] = True
        result["canonical_smiles"] = Chem.MolToSmiles(mol)
        result["properties"] = compute_molecule_properties(mol)

    return result


def compute_molecule_properties(mol) -> dict:
    """
    Compute key molecular properties using RDKit.

    Args:
        mol: RDKit molecule object.

    Returns:
        Dictionary of molecular properties.
    """
    return {
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 2),
        "logp": round(float(Descriptors.MolLogP(mol)), 2),
        "num_h_donors": int(Descriptors.NumHDonors(mol)),
        "num_h_acceptors": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": round(float(Descriptors.TPSA(mol)), 2),
        "num_rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "num_rings": int(Descriptors.RingCount(mol)),
        "qed_score": round(float(QED.qed(mol)), 4),
    }


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def evaluate_generated_molecules(
    generated_smiles: List[str],
    training_smiles: Optional[List[str]] = None,
) -> dict:
    """
    Evaluate generated molecules using full MOSES benchmark metrics.

    Metrics computed:
        - Validity, Uniqueness, Novelty (standard)
        - IntDiv1, IntDiv2 (internal diversity, MOSES)
        - Fragment similarity (BRICS fragments vs training set)
        - Scaffold similarity (Bemis-Murcko scaffolds vs training set)
        - SNN (similarity to nearest neighbor in training set)
        - Property distributions (MW, logP, QED, SA)

    References:
        Polykovskiy et al. 2020, "Molecular Sets (MOSES)"

    Args:
        generated_smiles: List of generated SMILES strings.
        training_smiles:  Original training SMILES (for novelty/similarity).

    Returns:
        Dictionary of evaluation metrics.
    """
    total_count = len(generated_smiles)

    # Validity: fraction that parse with RDKit
    valid_molecules = []
    valid_smiles_set = set()
    valid_mols = []  # RDKit mol objects for fingerprint computation
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            valid_molecules.append(canonical)
            valid_smiles_set.add(canonical)
            valid_mols.append(mol)

    validity_rate = len(valid_molecules) / total_count if total_count > 0 else 0.0

    # Uniqueness: fraction of valid molecules that are unique
    uniqueness_rate = len(valid_smiles_set) / len(valid_molecules) if valid_molecules else 0.0

    # Novelty: fraction of unique valid molecules NOT in training set
    novelty_rate = 0.0
    if training_smiles is not None and valid_smiles_set:
        training_set = set(training_smiles)
        novel_molecules = valid_smiles_set - training_set
        novelty_rate = len(novel_molecules) / len(valid_smiles_set)

    # Internal Diversity (IntDiv1 & IntDiv2) — MOSES metric
    intdiv1, intdiv2 = _compute_internal_diversity(list(valid_smiles_set)[:5000])

    # Fragment Similarity (BRICS) — MOSES metric
    frag_similarity = 0.0
    if training_smiles:
        frag_similarity = _compute_fragment_similarity(
            list(valid_smiles_set)[:5000], training_smiles[:5000]
        )

    # Scaffold Similarity — MOSES metric
    scaff_similarity = 0.0
    if training_smiles:
        scaff_similarity = _compute_scaffold_similarity(
            list(valid_smiles_set)[:5000], training_smiles[:5000]
        )

    # SNN (Similarity to Nearest Neighbor) — MOSES metric
    snn_score = 0.0
    if training_smiles:
        snn_score = _compute_snn(
            list(valid_smiles_set)[:1000], training_smiles[:10000]
        )

    metrics = {
        "total_generated": total_count,
        "valid_count": len(valid_molecules),
        "unique_count": len(valid_smiles_set),
        "validity_rate": round(float(validity_rate), 4),
        "uniqueness_rate": round(float(uniqueness_rate), 4),
        "novelty_rate": round(float(novelty_rate), 4),
        "intdiv1": round(float(intdiv1), 4),
        "intdiv2": round(float(intdiv2), 4),
        "frag_similarity": round(float(frag_similarity), 4),
        "scaff_similarity": round(float(scaff_similarity), 4),
        "snn": round(float(snn_score), 4),
    }

    logger.info(
        f"Evaluation: validity={metrics['validity_rate']:.2%}, "
        f"uniqueness={metrics['uniqueness_rate']:.2%}, "
        f"novelty={metrics['novelty_rate']:.2%}, "
        f"IntDiv1={metrics['intdiv1']:.4f}, "
        f"Frag={metrics['frag_similarity']:.4f}, "
        f"Scaff={metrics['scaff_similarity']:.4f}, "
        f"SNN={metrics['snn']:.4f}"
    )

    return metrics


def _compute_internal_diversity(smiles_list: List[str]) -> tuple:
    """
    Compute IntDiv1 and IntDiv2 (MOSES internal diversity metrics).

    IntDiv_p = 1 - (1/|G|^2 * sum T(m1,m2)^p)^(1/p)
    """
    if len(smiles_list) < 2:
        return 0.0, 0.0

    # Use new MorganGenerator API (replaces deprecated GetMorganFingerprintAsBitVect)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = morgan_gen.GetFingerprint(mol)
            fingerprints.append(fp)

    if len(fingerprints) < 2:
        return 0.0, 0.0

    # Sample pairs for efficiency (full computation is O(n^2))
    import random
    n = len(fingerprints)
    num_pairs = min(10000, n * (n - 1) // 2)

    sum_sim1 = 0.0
    sum_sim2 = 0.0
    for _ in range(num_pairs):
        i, j = random.sample(range(n), 2)
        sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        sum_sim1 += sim
        sum_sim2 += sim ** 2

    avg_sim1 = sum_sim1 / num_pairs
    avg_sim2 = (sum_sim2 / num_pairs) ** 0.5

    intdiv1 = 1.0 - avg_sim1
    intdiv2 = 1.0 - avg_sim2

    return intdiv1, intdiv2


def _compute_fragment_similarity(
    generated_smiles: List[str],
    reference_smiles: List[str],
) -> float:
    """
    MOSES Fragment Similarity: cosine similarity of BRICS fragment distributions.
    """
    from collections import Counter

    def get_fragments(smiles_list: List[str]) -> Counter:
        frag_counts: Counter = Counter()
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    frags = BRICS.BRICSDecompose(mol)
                    for f in frags:
                        frag_counts[f] += 1
                except Exception:
                    continue
        return frag_counts

    gen_frags = get_fragments(generated_smiles)
    ref_frags = get_fragments(reference_smiles)

    if not gen_frags or not ref_frags:
        return 0.0

    # Cosine similarity between fragment frequency vectors
    all_keys = set(gen_frags.keys()) | set(ref_frags.keys())
    dot_product = sum(gen_frags.get(k, 0) * ref_frags.get(k, 0) for k in all_keys)
    norm_gen = sum(v ** 2 for v in gen_frags.values()) ** 0.5
    norm_ref = sum(v ** 2 for v in ref_frags.values()) ** 0.5

    if norm_gen == 0 or norm_ref == 0:
        return 0.0

    return dot_product / (norm_gen * norm_ref)


def _compute_scaffold_similarity(
    generated_smiles: List[str],
    reference_smiles: List[str],
) -> float:
    """
    MOSES Scaffold Similarity: cosine similarity of Bemis-Murcko scaffold distributions.
    """
    from collections import Counter

    def get_scaffolds(smiles_list: List[str]) -> Counter:
        scaff_counts: Counter = Counter()
        for smi in smiles_list:
            try:
                scaff = MurckoScaffoldSmiles(smiles=smi)
                scaff_counts[scaff] += 1
            except Exception:
                continue
        return scaff_counts

    gen_scaff = get_scaffolds(generated_smiles)
    ref_scaff = get_scaffolds(reference_smiles)

    if not gen_scaff or not ref_scaff:
        return 0.0

    all_keys = set(gen_scaff.keys()) | set(ref_scaff.keys())
    dot_product = sum(gen_scaff.get(k, 0) * ref_scaff.get(k, 0) for k in all_keys)
    norm_gen = sum(v ** 2 for v in gen_scaff.values()) ** 0.5
    norm_ref = sum(v ** 2 for v in ref_scaff.values()) ** 0.5

    if norm_gen == 0 or norm_ref == 0:
        return 0.0

    return dot_product / (norm_gen * norm_ref)


def _compute_snn(
    generated_smiles: List[str],
    reference_smiles: List[str],
) -> float:
    """
    MOSES SNN: average Tanimoto similarity to nearest neighbor in reference set.
    """
    # Use new MorganGenerator API (replaces deprecated GetMorganFingerprintAsBitVect)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    # Compute fingerprints for reference set
    ref_fps = []
    for smi in reference_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = morgan_gen.GetFingerprint(mol)
            ref_fps.append(fp)

    if not ref_fps:
        return 0.0

    total_max_sim = 0.0
    count = 0
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        gen_fp = morgan_gen.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
        max_sim = max(sims) if sims else 0.0
        total_max_sim += max_sim
        count += 1

    return total_max_sim / count if count > 0 else 0.0
