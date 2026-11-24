"""
Graph VAE — RGCVAE-Inspired Variational Autoencoder for Molecular Graph Generation

Enhanced implementation integrating techniques from 5 research papers:

    1. Simonovsky & Komodakis (2018) "GraphVAE" — symmetric decoder, implicit
       node probabilities, weighted reconstruction loss
    2. Jin et al. (2018) "JT-VAE" — valence-constrained decoding concept
    3. De Cao & Kipf (2018) "MolGAN" — R-GCN edge-type awareness
    4. Nguyen & Karolak (2024) "TGVAE" — cyclical β annealing, global_add_pool
    5. Rigoni et al. (2025) "RGCVAE" — RGIN encoder, two-stage pairwise decoder,
       valence-masked decoding, rich atom features, per-bond-type MLPs

Architecture:
    Encoder: RGIN (Relational Graph Isomorphism Network) with residual connections
             + global_add_pool → μ, log σ²
    Latent:  Reparameterization trick (z = μ + ε·σ)
    Decoder: TWO-STAGE (Paper §3.2):
             Stage 1 — Node decoder:  z → per-node features + atom type logits
             Stage 2 — Edge decoder:  pairwise(node_i, node_j) → edge prob + bond type
             This matches the paper's atom decoding → edge decoding pipeline.

Key improvements over original GraphVAE:
    1. RGIN encoder with per-bond-type MLPs (vs bond-blind GCN)
    2. Rich atom features (28d vs 17d)
    3. global_add_pool (vs global_mean_pool)
    4. Cross-entropy node loss (vs MSE)
    5. TWO-STAGE pairwise decoder (vs flat MLP → reshape N×N)
    6. Teacher-forced bond loss (CE only where edges exist, Paper Eq. 16)
    7. Valence-masked generation (in graph_utils.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch

from app.models.graph_utils import NODE_FEATURE_DIM, NUM_ATOM_TYPES

# Bond types in molecular graphs
NUM_BOND_TYPES = 4  # single, double, triple, aromatic


# ---------------------------------------------------------------------------
# RGIN Convolution Layer (RGCVAE Paper §3.1, Eq. 2)
# ---------------------------------------------------------------------------

class RGINConv(nn.Module):
    """
    Relational Graph Isomorphism Network convolution layer.

    Extends GIN (Xu et al., 2019) with edge-type-specific MLPs, following
    RGCVAE (Rigoni et al., 2025, Eq. 2):

        h_v^(k) = MLP^(k)( (1 + ε^(k)) · h_v^(k-1)
                  + Σ_{u∈N(v)} MLP_{bond_type(v,u)}^(k)(h_u^(k-1)) )

    Each bond type (single, double, triple, aromatic) gets its own MLP
    to transform neighbor features. This is strictly more expressive than
    R-GCN (which uses linear transforms) while preserving GIN's equivalence
    to the Weisfeiler-Lehman graph isomorphism test.

    Ablation proof (RGCVAE Table 5):
        RGIN: 94.8% reconstruction on QM9
        R-GCN: 86.0%
        GIN (no edges): 41.6%
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_edge_types: int = NUM_BOND_TYPES,
        eps_init: float = 0.0,
    ):
        super().__init__()
        self.num_edge_types = num_edge_types

        # Learnable epsilon for self-loop weighting (GIN Theorem 3)
        self.eps = nn.Parameter(torch.tensor(eps_init))

        # Main MLP applied after aggregation (Eq. 2: outer MLP^(k))
        self.main_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Edge-type-specific MLPs for neighbor transformation
        # (Eq. 2: MLP^(k)_{Φe((v,u))} — one per bond type)
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, in_dim),
            )
            for _ in range(num_edge_types)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        RGIN forward pass.

        Args:
            x:          Node features   [num_nodes, in_dim]
            edge_index: Edge COO index  [2, num_edges]
            edge_type:  Bond type per edge [num_edges] (0-3)

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        src, dst = edge_index  # source → destination

        # Handle empty graphs (no edges)
        if edge_index.size(1) == 0:
            out = (1 + self.eps) * x
            return self.main_mlp(out)

        # Aggregate neighbor messages through bond-type-specific MLPs
        agg = torch.zeros_like(x)
        for t, mlp in enumerate(self.edge_mlps):
            mask = (edge_type == t)
            if mask.any():
                transformed = mlp(x[src[mask]])  # transform source nodes
                agg.index_add_(0, dst[mask], transformed)  # aggregate at destination

        # GIN update: (1 + ε) · self + aggregated neighbors
        out = (1 + self.eps) * x + agg
        return self.main_mlp(out)


# ---------------------------------------------------------------------------
# RGIN Encoder (replaces GCN encoder)
# ---------------------------------------------------------------------------

class RGINEncoder(nn.Module):
    """
    RGIN-based graph encoder with residual connections.

    Architecture:
        x → Linear(node_feature_dim, hidden_dim)  [initial embedding]
          → RGINConv → BN → ReLU + skip₁
          → RGINConv → BN → ReLU + skip₂
          → RGINConv → BN → ReLU + skip₃
          → global_add_pool → FC → μ, log σ²

    Changes from original GraphEncoder:
        - RGINConv instead of GCNConv (bond-aware, WL-equivalent)
        - global_add_pool instead of global_mean_pool (better for
          variable-size molecules; Papers 1, 4, 5 all agree)
        - Accepts edge_type as input for bond-type-specific processing
    """

    def __init__(
        self,
        node_feature_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
        num_edge_types: int = NUM_BOND_TYPES,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initial node embedding (dimension change: node_feature_dim → hidden_dim)
        self.node_embed = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # RGIN layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGINConv(hidden_dim, hidden_dim, num_edge_types)
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # LayerNorm after global_add_pool (stabilizes large summed values)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple:
        """
        Encode a batch of graphs into latent vectors.

        Args:
            x:          Node features   [total_nodes, node_feature_dim]
            edge_index: Edge COO index  [2, total_edges]
            edge_type:  Bond types      [total_edges] (0-3)
            batch:      Batch assignment [total_nodes]

        Returns:
            mu:     Latent means     [batch_size, latent_dim]
            logvar: Latent log-vars  [batch_size, latent_dim]
        """
        # Initial embedding
        h = self.node_embed(x)

        # RGIN layers with residual connections + dropout
        for conv, norm in zip(self.convs, self.norms):
            h_new = self.dropout(F.relu(norm(conv(h, edge_index, edge_type))))
            h = h_new + h  # residual connection

        # Global add pooling: per-graph representation
        h = global_add_pool(h, batch)  # [batch_size, hidden_dim]

        # LayerNorm: stabilize after sum pooling (values can be 30-40x larger)
        h = self.pool_norm(h)

        # Latent projections
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Clamp logvar to prevent exp() overflow in reparameterize
        logvar = logvar.clamp(-20.0, 20.0)

        return mu, logvar


# ---------------------------------------------------------------------------
# Two-Stage Pairwise Decoder (RGCVAE Paper §3.2)
# ---------------------------------------------------------------------------

class PairwiseDecoder(nn.Module):
    """
    Two-stage pairwise decoder matching RGCVAE Paper §3.2.

    CRITICAL CHANGE: Replaces the old flat MLP decoder (z → N×N matrix)
    which couldn't learn edge placement because a single 128-dim vector
    mapped to 1444 adjacency entries has no spatial structure.

    The new decoder follows the paper exactly:

    Stage 1 — Node Decoding (§3.2.1):
        z → MLP → per-node features r_v and atom type logits τ_v
        Each of the N atoms gets its own feature vector.

    Stage 2 — Edge Decoding (§3.2.2, Eq. 5-10):
        For each pair (v, u), compute permutation-invariant representation:
            ψ_{v,u} = Concat(s_v + s_u, s_v ⊙ s_u, Σ_i s_i)
        where s_v = Concat(r_v, W(τ_v))

        Then predict:
            P(v↔u)      = sigmoid(C(ψ_{v,u}))     — edge existence
            P(v-l-u)    = softmax(L_l(ψ_{v,u}))    — bond type

    Why this is better:
        - Edges are predicted from PAIRS of node features (local structure)
        - Sum/product operations capture complementary information
        - Global context Σ_i s_i gives molecule-wide awareness
        - Much more expressive than flat MLP for sparse adjacency
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        max_nodes: int = 38,
        node_feature_dim: int = NODE_FEATURE_DIM,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim

        # ------------------------------------------------------------------
        # Stage 1: Node decoder — z → per-node features + atom type logits
        # ------------------------------------------------------------------
        # Decode graph-level z into N node representations
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * hidden_dim),
        )

        # POSITIONAL EMBEDDING: gives each atom slot a unique identity.
        # Without this, all N node features come from the same z vector,
        # making the edge MLP unable to distinguish between atom pairs.
        # This is the key adaptation for graph-level z (vs paper's per-node z).
        self.pos_embed = nn.Embedding(max_nodes, hidden_dim)

        # Node feature prediction (atom type + continuous features)
        self.node_feature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim),
        )

        # NODE EXISTENCE HEAD — predicts whether each atom slot is real.
        # This eliminates disconnected "C.C.C..." fragments by learning
        # which slots should be active instead of relying on post-processing.
        self.node_existence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ------------------------------------------------------------------
        # Stage 2: Edge decoder — pairwise predictions (Paper §3.2.2)
        # ------------------------------------------------------------------
        # Pairwise representation dimension:
        #   ψ = Concat(s_v + s_u, s_v ⊙ s_u, Σ s_i)
        #   = hidden_dim + hidden_dim + hidden_dim = 3 * hidden_dim
        pairwise_dim = 3 * hidden_dim

        # Edge existence predictor C(ψ) → scalar logit (Paper Eq. 9)
        self.edge_mlp = nn.Sequential(
            nn.Linear(pairwise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Bond type predictor L_l(ψ) → NUM_BOND_TYPES logits (Paper Eq. 10)
        self.bond_mlp = nn.Sequential(
            nn.Linear(pairwise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_BOND_TYPES),
        )

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Two-stage decode: node features → pairwise edge prediction.

        Args:
            z: Latent vectors [B, latent_dim]

        Returns:
            adj_logits:        Edge existence logits [B, N, N] (symmetric, diag=0)
            node_pred:         Node feature predictions [B, N, node_feature_dim]
            bond_pred:         Bond type logits [B, N, N, NUM_BOND_TYPES]
            node_exist_logits: Node existence logits [B, N] (learned, not heuristic)
        """
        B = z.size(0)
        N = self.max_nodes
        device = z.device

        # ==== Stage 1: Node Decoding ====
        # z → [B, N, hidden_dim] per-node features
        node_hidden = self.node_decoder(z)                     # [B, N*H]
        node_hidden = node_hidden.view(B, N, self.hidden_dim)  # [B, N, H]

        # Add positional embedding: each atom slot gets a unique identity
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        node_hidden = node_hidden + self.pos_embed(positions)  # [B, N, H]

        # Predict node features (atom types + continuous)
        node_pred = self.node_feature_head(node_hidden)  # [B, N, node_feature_dim]

        # Predict node existence (learned head, replaces heuristic max-edge)
        node_exist_logits = self.node_existence_head(node_hidden).squeeze(-1)  # [B, N]

        # ==== Stage 2: Edge Decoding (Paper §3.2.2) ====
        # s_v = node_hidden[v] (already contains atom info)
        s = node_hidden  # [B, N, H]

        # Global context: Σ_i s_i (molecule-wide sum, Paper Eq. 6)
        s_global = s.sum(dim=1, keepdim=True).expand_as(s)  # [B, N, H]

        # Pre-compute all pairwise representations efficiently
        # For each pair (i, j):
        #   ψ_{i,j} = Concat(s_i + s_j, s_i ⊙ s_j, Σ s_k)
        # Use broadcasting to compute all N×N pairs at once
        s_i = s.unsqueeze(2).expand(B, N, N, self.hidden_dim)  # [B, N, 1, H] → [B, N, N, H]
        s_j = s.unsqueeze(1).expand(B, N, N, self.hidden_dim)  # [B, 1, N, H] → [B, N, N, H]
        s_g = s_global.unsqueeze(2).expand(B, N, N, self.hidden_dim)  # [B, N, N, H]

        # Permutation-invariant pairwise representation (Paper Eq. 6)
        psi = torch.cat([
            s_i + s_j,    # additive interaction (linear)
            s_i * s_j,    # multiplicative interaction (non-linear)
            s_g,          # global molecule context
        ], dim=-1)        # [B, N, N, 3*H]

        # Edge existence logits: C(ψ) → [B, N, N, 1]
        adj_logits = self.edge_mlp(psi).squeeze(-1)  # [B, N, N]

        # Make symmetric: (logits + logits^T) / 2
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2.0

        # Zero diagonal (no self-loops, Paper mask M^e)
        diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        adj_logits = adj_logits.masked_fill(diag_mask, 0.0)

        # Bond type logits: L(ψ) → [B, N, N, NUM_BOND_TYPES]
        bond_pred = self.bond_mlp(psi)  # [B, N, N, NUM_BOND_TYPES]

        # Make bond types symmetric too
        bond_pred = (bond_pred + bond_pred.transpose(1, 2)) / 2.0

        return adj_logits, node_pred, bond_pred, node_exist_logits


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------

# V2 added a learned node-existence head to the decoder. Checkpoints trained
# before that ("V1") carry no weights for it, and the mismatch is missing keys
# only — no shape conflict — so a V1 checkpoint loads into a V2 model without
# error and leaves the head at random init. Node existence decides which atom
# slots become atoms, so generation would then sample noise and report nothing
# wrong. Detect the head and fall back to the V1 heuristic when it is absent.
NODE_EXISTENCE_KEY_PREFIX = "decoder.node_existence_head."


def checkpoint_has_node_existence_head(state_dict: dict) -> bool:
    """True if `state_dict` carries trained node-existence weights (V2)."""
    return any(key.startswith(NODE_EXISTENCE_KEY_PREFIX) for key in state_dict)


# ---------------------------------------------------------------------------
# Full GraphVAE (RGCVAE-inspired)
# ---------------------------------------------------------------------------

class GraphVAE(nn.Module):
    """
    RGCVAE-Inspired Graph Variational Autoencoder for molecular generation.

    Enhanced with techniques from 5 research papers:
        1. RGIN encoder (bond-type-specific MLPs, WL-equivalent)
        2. Rich 28-dim atom features (type + valence + charge + chirality + hybrid)
        3. global_add_pool (better for variable-size molecules)
        4. Cross-entropy node loss (correct for categorical atom types)
        5. TWO-STAGE pairwise decoder (Paper §3.2)
        6. Implicit node probabilities
        7. Bond type prediction
        8. Teacher-forced bond loss
        9. Valence-masked generation (in graph_utils.py)
        10. Cyclical β annealing (in training service)

    Usage:
        model = GraphVAE(max_nodes=38, latent_dim=128, hidden_dim=256)
        (adj_pred, node_pred, bond_pred,
         mu, logvar, adj_logits, node_exist_logits) = model(batch)
        loss = compute_graph_vae_loss(adj_pred, node_pred, bond_pred, ...)
    """

    def __init__(
        self,
        max_nodes: int = 38,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        node_feature_dim: int = NODE_FEATURE_DIM,
        num_encoder_layers: int = 3,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim

        # RGIN encoder (replaces GCN — Paper 5)
        self.encoder = RGINEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_encoder_layers,
        )

        # Two-stage pairwise decoder (Paper §3.2)
        self.decoder = PairwiseDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            node_feature_dim=node_feature_dim,
        )

        # Whether decoder.node_existence_head holds trained weights. The
        # checkpoint loader clears this for V1 checkpoints so that forward and
        # generate use the V1 heuristic instead of a randomly initialised head
        # (see checkpoint_has_node_existence_head).
        self.use_learned_node_existence = True

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·σ"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # use mean for inference

    def _node_existence(
        self, adj_pred: torch.Tensor, node_exist_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-slot node existence probabilities [B, max_nodes].

        V2 uses the learned head. V1 checkpoints have no trained head, so they
        fall back to the original heuristic — a slot is as "real" as its
        strongest edge — which reproduces their documented behaviour rather
        than reading a random head.
        """
        if self.use_learned_node_existence:
            return torch.sigmoid(node_exist_logits)
        return adj_pred.max(dim=2).values

    def forward(self, batch: Batch) -> tuple:
        """
        Full forward pass: encode → reparameterize → decode.

        Args:
            batch: PyG Batch object with x, edge_index, edge_type, batch attributes.

        Returns:
            adj_pred:   [batch_size, max_nodes, max_nodes] (probabilities with node existence)
            node_pred:  [batch_size, max_nodes, node_feature_dim]
            bond_pred:  [batch_size, max_nodes, max_nodes, NUM_BOND_TYPES]
            mu:                [batch_size, latent_dim]
            logvar:            [batch_size, latent_dim]
            adj_logits:        [batch_size, max_nodes, max_nodes] (raw logits for loss)
            node_exist_logits: [batch_size, max_nodes] (raw logits for loss)
        """
        # RGIN encoder requires edge_type
        mu, logvar = self.encoder(
            batch.x, batch.edge_index, batch.edge_type, batch.batch
        )
        z = self.reparameterize(mu, logvar)
        adj_logits, node_pred, bond_pred, node_exist_logits = self.decoder(z)

        # Build adj_pred (probabilities) with node existence on the diagonal
        adj_pred = torch.sigmoid(adj_logits)
        node_existence = self._node_existence(adj_pred, node_exist_logits)  # [B, k]
        adj_pred = adj_pred + torch.diag_embed(node_existence)  # set diagonal

        return adj_pred, node_pred, bond_pred, mu, logvar, adj_logits, node_exist_logits

    def generate(
        self,
        num_molecules: int = 10,
        device: torch.device = None,
        temperature: float = 1.0,
    ) -> tuple:
        """
        Generate new graphs by sampling from the latent space.

        Args:
            num_molecules: Number of molecules to generate.
            device:        Target device.
            temperature:   Sampling temperature (higher = more diverse).

        Returns:
            adj_pred:  [num_molecules, max_nodes, max_nodes]
            node_pred: [num_molecules, max_nodes, node_feature_dim]
            bond_pred: [num_molecules, max_nodes, max_nodes, NUM_BOND_TYPES]
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # BATCHED GENERATION: pairwise decoder creates [B, 38, 38, 768] tensor.
        # B=10000 → ~41 GB OOM. Chunk into batches of 128, concat on CPU.
        gen_batch_size = 128
        all_adj, all_node, all_bond = [], [], []

        with torch.no_grad():
            for start in range(0, num_molecules, gen_batch_size):
                end = min(start + gen_batch_size, num_molecules)
                z = torch.randn(end - start, self.latent_dim, device=device)
                z = z * temperature
                adj_logits, node_pred_b, bond_pred_b, node_exist_logits = self.decoder(z)

                adj_pred_b = torch.sigmoid(adj_logits)
                # Learned head for V2 checkpoints, max-edge heuristic for V1
                node_existence = self._node_existence(adj_pred_b, node_exist_logits)
                adj_pred_b = adj_pred_b + torch.diag_embed(node_existence)

                all_adj.append(adj_pred_b.cpu())
                all_node.append(node_pred_b.cpu())
                all_bond.append(bond_pred_b.cpu())

        adj_pred  = torch.cat(all_adj,  dim=0).to(device)
        node_pred = torch.cat(all_node, dim=0).to(device)
        bond_pred = torch.cat(all_bond, dim=0).to(device)

        return adj_pred, node_pred, bond_pred


# ---------------------------------------------------------------------------
# Loss Functions (RGCVAE Paper §4, Eq. 13-18)
# ---------------------------------------------------------------------------

def compute_graph_vae_loss(
    adj_pred: torch.Tensor,
    node_pred: torch.Tensor,
    bond_pred: torch.Tensor,
    adj_true: torch.Tensor,
    node_true: torch.Tensor,
    bond_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    adj_logits: torch.Tensor = None,
    node_exist_logits: torch.Tensor = None,
    kl_weight: float = 0.005,
    lambda_adj: float = 1.0,
    lambda_node: float = 1.0,
    lambda_bond: float = 1.0,
    lambda_exist: float = 2.0,
) -> dict:
    """
    Compute the RGCVAE loss (Paper §4, Eq. 13-18) with node existence.

    Loss = λ_A * L_edge + λ_exist * L_exist + λ_node * L_node
         + λ_bond * L_bond + β * L_KL

    Key additions:
        - L_exist: Learned node existence BCE (replaces heuristic diagonal).
          Weighted at λ=2.0 to strongly incentivize correct atom count.
    """
    k = adj_pred.size(1)
    mask_diag = torch.eye(k, device=adj_pred.device, dtype=torch.bool)
    mask_off = ~mask_diag

    # --- L_edge: Adjacency reconstruction (Paper Eq. 17) ---
    # pos_weight compensates for edge sparsity with graph-level z.
    off_diag_true = adj_true[:, mask_off].float()
    num_edges = off_diag_true.sum()
    num_total = off_diag_true.numel()
    num_non_edges = num_total - num_edges
    pw = (num_non_edges / (num_edges + 1e-8)).clamp(max=25.0)

    if adj_logits is not None:
        edge_exist_loss = F.binary_cross_entropy_with_logits(
            adj_logits[:, mask_off],
            off_diag_true,
            pos_weight=pw,
            reduction="mean",
        )
    else:
        p = adj_pred[:, mask_off]
        p = torch.nan_to_num(p, nan=0.5).clamp(1e-7, 1 - 1e-7)
        t = off_diag_true.clamp(0, 1)
        edge_exist_loss = -(pw * t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()

    # --- L_exist: Node existence from LEARNED HEAD ---
    # Supervised BCE: target = 1 for real atoms, 0 for padding.
    # The diagonal of adj_true encodes node existence (1 = real atom).
    node_exist_target = adj_true[:, mask_diag].float().clamp(0, 1)  # [B, N]

    if node_exist_logits is not None:
        # Learned head — use logits directly with BCE
        node_exist_loss = F.binary_cross_entropy_with_logits(
            node_exist_logits, node_exist_target, reduction="mean"
        )
    else:
        # Fallback: use diagonal of adj_pred (legacy)
        p_diag = adj_pred[:, mask_diag]
        p_diag = torch.nan_to_num(p_diag, nan=0.5).clamp(1e-7, 1 - 1e-7)
        node_exist_loss = -(
            node_exist_target * torch.log(p_diag)
            + (1 - node_exist_target) * torch.log(1 - p_diag)
        ).mean()

    # --- L_node: Node feature reconstruction (Paper Eq. 15) ---
    # Cross-entropy for categorical atom types
    node_logits = node_pred[:, :, :NUM_ATOM_TYPES]     # [B, N, NUM_ATOM_TYPES]
    node_targets = node_true[:, :, :NUM_ATOM_TYPES].argmax(dim=-1)  # [B, N]

    B, N = node_targets.shape
    recon_node_loss = F.cross_entropy(
        node_logits.reshape(B * N, NUM_ATOM_TYPES),
        node_targets.reshape(B * N),
        reduction="mean",
    )

    # Also MSE for continuous features (valence, charge, etc.)
    if node_pred.size(2) > NUM_ATOM_TYPES:
        continuous_pred = node_pred[:, :, NUM_ATOM_TYPES:]
        continuous_true = node_true[:, :, NUM_ATOM_TYPES:]
        recon_continuous_loss = F.mse_loss(
            continuous_pred, continuous_true, reduction="mean"
        )
        recon_node_loss = recon_node_loss + 0.5 * recon_continuous_loss

    # --- L_bond: Bond type reconstruction — TEACHER FORCED (Paper Eq. 16) ---
    B_sz, N_sz, _, C = bond_pred.shape
    adj_mask = adj_true[:, :, :].float()  # [B, N, N] — 1 where edge exists
    num_edges_bond = adj_mask.sum()

    if num_edges_bond > 0:
        edge_positions = adj_mask.bool()
        bond_logits_at_edges = bond_pred[edge_positions]
        bond_targets_at_edges = bond_true[edge_positions].long()
        recon_bond_loss = F.cross_entropy(
            bond_logits_at_edges, bond_targets_at_edges, reduction="mean"
        )
    else:
        recon_bond_loss = torch.tensor(0.0, device=adj_pred.device)

    # --- L_KL: KL divergence with FREE BITS (Paper Eq. 18) ---
    FREE_BITS = 0.25
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=FREE_BITS)
    kl_loss = kl_per_dim.sum(dim=1).mean()

    # --- Weighted total loss ---
    total_loss = (
        lambda_adj * edge_exist_loss
        + lambda_exist * node_exist_loss
        + lambda_node * recon_node_loss
        + lambda_bond * recon_bond_loss
        + kl_weight * kl_loss
    )

    return {
        "total_loss": total_loss,
        "recon_adj_loss": edge_exist_loss,
        "node_exist_loss": node_exist_loss,
        "recon_node_loss": recon_node_loss,
        "recon_bond_loss": recon_bond_loss,
        "kl_loss": kl_loss,
    }
