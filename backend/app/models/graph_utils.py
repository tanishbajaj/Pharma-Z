"""
Graph Utilities — SMILES ↔ Molecular Graph Conversion

Converts SMILES strings to PyTorch Geometric Data objects and back.
Uses RDKit for molecule parsing and property extraction.

Enhanced with:
    - Rich atom features (28d) including valence, charge, chirality,
      hybridization — based on RGCVAE (Rigoni et al., 2025)
    - Edge type extraction in COO format for RGIN encoder
    - Bond type extraction and MST connectivity enforcement
      based on Simonovsky & Komodakis (2018) "GraphVAE"
    - Valence-masked molecule generation (RGCVAE + JT-VAE)
"""

from typing import Optional, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdmolops, rdchem


# Bond type mapping (Paper §3.2: de=4 bond types)
BOND_TYPE_MAP = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
INV_BOND_TYPE_MAP = {v: k for k, v in BOND_TYPE_MAP.items()}
NUM_BOND_TYPES = len(BOND_TYPE_MAP)


# ---------------------------------------------------------------------------
# Atom Feature Encoding (RGCVAE-enhanced: 28 dimensions)
# ---------------------------------------------------------------------------

# Common atoms in drug-like molecules (covers >99% of ZINC)
ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
ATOM_TYPE_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES) + 1  # +1 for "other"

MAX_DEGREE = 5
DEGREE_DIM = MAX_DEGREE + 1  # 0..5

# Chirality one-hot (4 tags: unspecified, CW, CCW, other)
CHIRAL_DIM = 4

# Hybridization one-hot (SP, SP2, SP3)
HYBRID_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
]
HYBRID_DIM = len(HYBRID_TYPES)

# Maximum atom valence in training data (for normalization)
MAX_ATOM_VALENCE = 8

# Atom valence lookup for valence-masked generation
ATOM_VALENCE_MAP = {
    6: 4,   # Carbon
    7: 3,   # Nitrogen
    8: 2,   # Oxygen
    9: 1,   # Fluorine
    15: 5,  # Phosphorus
    16: 6,  # Sulfur
    17: 1,  # Chlorine
    35: 1,  # Bromine
    53: 1,  # Iodine
}

# Total node feature dimension (RGCVAE representation 3):
# atom_type (10) + degree (6) + aromatic (1)
# + valence (1) + charge (1) + num_Hs (1) + in_ring (1)
# + chirality (4) + hybridization (3) = 28
NODE_FEATURE_DIM = (
    NUM_ATOM_TYPES   # 10: atom type one-hot
    + DEGREE_DIM     # 6:  degree one-hot
    + 1              # 1:  is aromatic
    + 1              # 1:  total valence (normalized)
    + 1              # 1:  formal charge (normalized)
    + 1              # 1:  num hydrogens (normalized)
    + 1              # 1:  is in ring
    + CHIRAL_DIM     # 4:  chirality one-hot
    + HYBRID_DIM     # 3:  hybridization one-hot
)  # = 28


def atom_to_features(atom) -> List[float]:
    """
    Convert an RDKit atom to a rich feature vector (RGCVAE-enhanced).

    Features (28 dimensions total):
        - Atom type:      one-hot  (10 dim) — C, N, O, F, P, S, Cl, Br, I, Other
        - Degree:         one-hot  (6 dim)  — 0, 1, 2, 3, 4, 5
        - Aromatic:       binary   (1 dim)  — 0 or 1
        - Total valence:  scalar   (1 dim)  — normalized by MAX_ATOM_VALENCE
        - Formal charge:  scalar   (1 dim)  — normalized by 4
        - Num hydrogens:  scalar   (1 dim)  — normalized by MAX_ATOM_VALENCE
        - Is in ring:     binary   (1 dim)  — 0 or 1
        - Chirality:      one-hot  (4 dim)  — unspecified, CW, CCW, other
        - Hybridization:  one-hot  (3 dim)  — SP, SP2, SP3

    Based on RGCVAE (Rigoni et al., 2025) representation 3:
        atom type + total valence + formal charge + chiral property.
    Extended with hybridization and ring membership from JT-VAE
    (Jin et al., 2018) for even richer encoding.

    Returns:
        List of floats (length NODE_FEATURE_DIM = 28)
    """
    features = []

    # 1. Atom type: one-hot (10 dim)
    atom_type = [0.0] * NUM_ATOM_TYPES
    atomic_num = atom.GetAtomicNum()
    idx = ATOM_TYPE_TO_IDX.get(atomic_num, NUM_ATOM_TYPES - 1)
    atom_type[idx] = 1.0
    features.extend(atom_type)

    # 2. Degree: one-hot (6 dim)
    degree = [0.0] * DEGREE_DIM
    deg = min(atom.GetDegree(), MAX_DEGREE)
    degree[deg] = 1.0
    features.extend(degree)

    # 3. Aromatic flag (1 dim)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # --- NEW features from RGCVAE / JT-VAE ---

    # 4. Total valence, normalized (1 dim)
    features.append(atom.GetTotalValence() / MAX_ATOM_VALENCE)

    # 5. Formal charge, normalized (1 dim)
    features.append(atom.GetFormalCharge() / 4.0)

    # 6. Number of hydrogens, normalized (1 dim)
    features.append(atom.GetTotalNumHs() / MAX_ATOM_VALENCE)

    # 7. Is in ring (1 dim)
    features.append(1.0 if atom.IsInRing() else 0.0)

    # 8. Chirality: one-hot (4 dim)
    chiral = int(atom.GetChiralTag())
    chiral_oh = [0.0] * CHIRAL_DIM
    if 0 <= chiral < CHIRAL_DIM:
        chiral_oh[chiral] = 1.0
    features.extend(chiral_oh)

    # 9. Hybridization: one-hot (3 dim)
    hyb = atom.GetHybridization()
    hyb_oh = [0.0] * HYBRID_DIM
    if hyb in HYBRID_TYPES:
        hyb_oh[HYBRID_TYPES.index(hyb)] = 1.0
    features.extend(hyb_oh)

    return features


# ---------------------------------------------------------------------------
# SMILES → Graph
# ---------------------------------------------------------------------------

def smiles_to_graph(
    smiles: str,
    max_nodes: int = 38,
) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles:    SMILES representation of the molecule.
        max_nodes: Maximum number of atoms allowed. Molecules with more
                   atoms are skipped (returns None).

    Returns:
        Data object with:
            x:         Node feature matrix    [num_atoms, NODE_FEATURE_DIM]
            edge_index: COO edge index        [2, num_edges]
            adj:       Dense adjacency matrix [max_nodes, max_nodes]
            num_nodes: Actual number of atoms
            smiles:    Original SMILES string
        None if the molecule is invalid or exceeds max_nodes.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add explicit hydrogens? No — keep implicit for smaller graphs.
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0 or num_atoms > max_nodes:
        return None

    # --- Node features (RGCVAE-enhanced: 28 dim) ---
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_to_features(atom))

    x = torch.tensor(node_features, dtype=torch.float32)

    # --- Edge index (COO format, undirected) ---
    edge_list: List[Tuple[int, int]] = []
    edge_types_list: List[int] = []  # NEW: bond type per edge for RGIN encoder
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        edge_list.append((i, j))
        edge_types_list.append(bt)
        edge_list.append((j, i))  # undirected
        edge_types_list.append(bt)

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types_list, dtype=torch.long)  # [num_edges]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    # --- Dense adjacency matrix (padded to max_nodes) ---
    adj = torch.zeros(max_nodes, max_nodes, dtype=torch.float32)
    # Set diagonal for active nodes (implicit node existence)
    for atom_idx in range(num_atoms):
        adj[atom_idx, atom_idx] = 1.0

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # --- Bond type matrix (padded to max_nodes) ---
    # 0 = no bond, 1 = single, 2 = double, 3 = triple, 4 = aromatic
    # (shifted by +1 so 0 can mean "no bond" for cross-entropy target)
    bond_types = torch.zeros(max_nodes, max_nodes, dtype=torch.long)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        bond_types[i, j] = bt
        bond_types[j, i] = bt

    # --- Node mask: which nodes are real (not padding) ---
    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    node_mask[:num_atoms] = True

    # --- Build Data object ---
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,       # NEW: bond types in COO format for RGIN
        adj=adj,
        bond_types=bond_types,
        node_mask=node_mask,
        num_nodes=num_atoms,
        smiles=smiles,
    )

    return data


# ---------------------------------------------------------------------------
# Graph → SMILES (best-effort reconstruction)
# ---------------------------------------------------------------------------

def adjacency_to_smiles(
    adj_matrix: np.ndarray,
    node_types: Optional[np.ndarray] = None,
    bond_types: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    enforce_connected: bool = True,
) -> Optional[str]:
    """
    Convert a predicted adjacency matrix back to a SMILES string.

    Enhanced reconstruction (Paper §4.1):
        1. Use diagonal for node existence (implicit node probabilities).
        2. Threshold off-diagonal for edges.
        3. If enforce_connected, add MST edges to ensure connectivity.
        4. Assign atom types from node_types (or default to Carbon).
        5. Assign bond types from bond_types (or default to SINGLE).
        6. Build an RDKit molecule, sanitize, return canonical SMILES.

    Args:
        adj_matrix:        Predicted adjacency matrix [max_nodes, max_nodes].
        node_types:        Optional array of atom type indices [max_nodes].
        bond_types:        Optional array of bond type indices [max_nodes, max_nodes].
        threshold:         Threshold for deciding if a node/edge exists.
        enforce_connected: Use MST to ensure molecule is connected (Paper §4.1).

    Returns:
        Canonical SMILES string, or None if reconstruction fails.
    """
    try:
        n = adj_matrix.shape[0]

        # Node existence from diagonal (implicit node probabilities)
        node_probs = np.diag(adj_matrix)
        active_mask = node_probs > threshold

        if not active_mask.any():
            return None

        active_indices = np.where(active_mask)[0]
        if len(active_indices) < 2:
            return None

        # Edge existence from off-diagonal (thresholded)
        edge_adj = adj_matrix.copy()
        np.fill_diagonal(edge_adj, 0)  # remove diagonal
        edge_adj = np.maximum(edge_adj, edge_adj.T)  # ensure symmetric
        binary_adj = (edge_adj > threshold).astype(int)

        # Enforce connectivity via MST (Paper §4.1)
        if enforce_connected:
            binary_adj = enforce_graph_connectivity(
                binary_adj, edge_adj, active_indices
            )

        # Build RWMol
        mol = Chem.RWMol()

        # Add atoms
        idx_map = {}
        for i, orig_idx in enumerate(active_indices):
            if node_types is not None:
                atom_type_idx = int(node_types[orig_idx])
                if 0 <= atom_type_idx < len(ATOM_TYPES):
                    atomic_num = ATOM_TYPES[atom_type_idx]
                else:
                    atomic_num = 6  # default Carbon
            else:
                atomic_num = 6  # default Carbon

            atom = Chem.Atom(atomic_num)
            new_idx = mol.AddAtom(atom)
            idx_map[orig_idx] = new_idx

        # Add bonds with predicted bond types
        added_bonds = set()
        for orig_i in active_indices:
            for orig_j in active_indices:
                if orig_i >= orig_j:
                    continue
                if binary_adj[orig_i, orig_j] == 1:
                    bond_key = (idx_map[orig_i], idx_map[orig_j])
                    if bond_key not in added_bonds:
                        # Determine bond type
                        if bond_types is not None:
                            bt_idx = int(bond_types[orig_i, orig_j])
                            rdkit_bt = INV_BOND_TYPE_MAP.get(
                                bt_idx, Chem.BondType.SINGLE
                            )
                        else:
                            rdkit_bt = Chem.BondType.SINGLE

                        mol.AddBond(bond_key[0], bond_key[1], rdkit_bt)
                        added_bonds.add(bond_key)

        # Sanitize
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            # Verify round-trip
            if Chem.MolFromSmiles(smiles) is not None:
                return smiles
        except Exception:
            pass

        return None

    except Exception:
        return None


def adjacency_to_smiles_valence_masked(
    adj_matrix: np.ndarray,
    node_features: np.ndarray,
    bond_logits: np.ndarray,
    threshold: float = 0.5,
) -> Optional[str]:
    """
    Convert predicted graph outputs to SMILES using valence-masked
    iterative edge placement (RGCVAE + JT-VAE approach).

    Unlike threshold-based reconstruction, this method:
        1. Picks atom types via argmax
        2. Sorts candidate edges by probability (descending)
        3. Iteratively places edges, checking valence budget at each step
        4. For each edge, picks the highest-probability bond type that
           respects both atoms' remaining valence
        5. Adds implicit hydrogens to fill remaining valence

    This enforces chemical validity during generation, achieving ~99.9%
    validity (RGCVAE, Rigoni et al. 2025) vs ~55% with naive thresholding.

    Args:
        adj_matrix:    Predicted adjacency [max_nodes, max_nodes].
                       Diagonal = node existence probabilities.
                       Off-diagonal = edge existence probabilities.
        node_features: Predicted node features [max_nodes, NODE_FEATURE_DIM].
                       First NUM_ATOM_TYPES dims are atom type logits.
        bond_logits:   Predicted bond type logits [max_nodes, max_nodes, NUM_BOND_TYPES].
        threshold:     Threshold for node existence (diagonal of adj_matrix).

    Returns:
        Canonical SMILES string, or None if reconstruction fails.
    """
    try:
        n = adj_matrix.shape[0]

        # --- Step 1: Determine active nodes from diagonal ---
        node_probs = np.diag(adj_matrix)
        active_mask = node_probs > threshold
        if not active_mask.any():
            return None

        active_indices = np.where(active_mask)[0]
        if len(active_indices) < 2:
            return None

        # --- Step 2: Assign atom types via argmax ---
        atom_type_logits = node_features[:, :NUM_ATOM_TYPES]
        atom_type_indices = np.argmax(atom_type_logits, axis=1)

        # Map to atomic numbers and get max valence for each atom
        atomic_nums = {}
        max_valences = {}
        for idx in active_indices:
            at_idx = int(atom_type_indices[idx])
            if 0 <= at_idx < len(ATOM_TYPES):
                anum = ATOM_TYPES[at_idx]
            else:
                anum = 6  # default Carbon
            atomic_nums[idx] = anum
            max_valences[idx] = ATOM_VALENCE_MAP.get(anum, 4)

        # Track remaining valence for each atom
        remaining_valence = {idx: max_valences[idx] for idx in active_indices}

        # --- Step 3: Collect candidate edges and sort by probability ---
        edge_adj = adj_matrix.copy()
        np.fill_diagonal(edge_adj, 0)
        edge_adj = np.maximum(edge_adj, edge_adj.T)  # symmetrize

        candidates = []
        for i_pos, orig_i in enumerate(active_indices):
            for j_pos in range(i_pos + 1, len(active_indices)):
                orig_j = active_indices[j_pos]
                prob = edge_adj[orig_i, orig_j]
                if prob > 0.1:  # only consider edges with some probability
                    candidates.append((prob, orig_i, orig_j))

        # Sort by probability (descending) — most confident edges first
        candidates.sort(key=lambda x: x[0], reverse=True)

        # --- Step 4: Iteratively place edges with valence checking ---
        bond_weight_map = {0: 1, 1: 2, 2: 3, 3: 1}  # single=1, double=2, triple=3, aromatic=1

        placed_bonds = []  # List of (i, j, bond_type_idx)

        for prob, orig_i, orig_j in candidates:
            if prob < threshold:
                continue  # below edge existence threshold

            # Get bond type probabilities for this edge
            bt_probs = bond_logits[orig_i, orig_j]  # [NUM_BOND_TYPES]
            bt_order = np.argsort(bt_probs)[::-1]  # highest probability first

            # Try each bond type, pick the first that respects both atoms' valence
            placed = False
            for bt_idx in bt_order:
                bw = bond_weight_map.get(int(bt_idx), 1)
                if remaining_valence[orig_i] >= bw and remaining_valence[orig_j] >= bw:
                    placed_bonds.append((orig_i, orig_j, int(bt_idx)))
                    remaining_valence[orig_i] -= bw
                    remaining_valence[orig_j] -= bw
                    placed = True
                    break

            # If no bond type fits, skip this edge entirely

        # --- Step 5: Build RDKit molecule ---
        mol = Chem.RWMol()

        # Add atoms
        idx_map = {}
        for orig_idx in active_indices:
            atom = Chem.Atom(atomic_nums[orig_idx])
            new_idx = mol.AddAtom(atom)
            idx_map[orig_idx] = new_idx

        # Add bonds
        for orig_i, orig_j, bt_idx in placed_bonds:
            rdkit_bt = INV_BOND_TYPE_MAP.get(bt_idx, Chem.BondType.SINGLE)
            mol.AddBond(idx_map[orig_i], idx_map[orig_j], rdkit_bt)

        # --- Step 6: Sanitize and return ---
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            # Verify round-trip
            if Chem.MolFromSmiles(smiles) is not None:
                return smiles
        except Exception:
            # Try without sanitization flags that might fail
            try:
                Chem.SanitizeMol(
                    mol,
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                )
                smiles = Chem.MolToSmiles(mol)
                if Chem.MolFromSmiles(smiles) is not None:
                    return smiles
            except Exception:
                pass

        return None

    except Exception:
        return None


def enforce_graph_connectivity(
    binary_adj: np.ndarray,
    edge_weights: np.ndarray,
    active_indices: np.ndarray,
) -> np.ndarray:
    """
    Ensure generated graph is connected via maximum spanning tree (Paper §4.1).

    "We use prior knowledge that molecules are connected and, at test time only,
     construct maximum spanning tree on the set of probable nodes in order to
     include its edges even if A_a,b < 0.5 originally." — Paper §4.1

    Args:
        binary_adj:     Thresholded binary adjacency [N, N].
        edge_weights:   Raw edge probabilities [N, N] (used as MST weights).
        active_indices: Indices of nodes deemed active.

    Returns:
        Updated binary adjacency with MST edges added for connectivity.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix

    n_active = len(active_indices)
    if n_active < 2:
        return binary_adj

    # Build a sub-adjacency of active nodes with NEGATIVE weights
    # (scipy finds MINIMUM spanning tree, so negate for maximum)
    sub_weights = np.zeros((n_active, n_active))
    for i_idx, orig_i in enumerate(active_indices):
        for j_idx, orig_j in enumerate(active_indices):
            if i_idx < j_idx:
                w = edge_weights[orig_i, orig_j]
                sub_weights[i_idx, j_idx] = -w  # negate for max spanning tree
                sub_weights[j_idx, i_idx] = -w

    # Compute MST
    sparse_sub = csr_matrix(np.triu(sub_weights))
    mst = minimum_spanning_tree(sparse_sub)
    mst_dense = mst.toarray()

    # Add MST edges to the binary adjacency
    result = binary_adj.copy()
    for i_idx in range(n_active):
        for j_idx in range(i_idx + 1, n_active):
            if mst_dense[i_idx, j_idx] != 0:
                orig_i = active_indices[i_idx]
                orig_j = active_indices[j_idx]
                result[orig_i, orig_j] = 1
                result[orig_j, orig_i] = 1

    return result


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

def smiles_list_to_graphs(
    smiles_list: List[str],
    max_nodes: int = 38,
    show_progress: bool = True,
) -> Tuple[List[Data], int, int]:
    """
    Convert a list of SMILES to a list of graph Data objects.

    Args:
        smiles_list: List of SMILES strings.
        max_nodes:   Maximum atoms per molecule.
        show_progress: Print conversion progress.

    Returns:
        Tuple of (graph_list, valid_count, skipped_count).
    """
    graphs: List[Data] = []
    skipped = 0

    for i, smiles in enumerate(smiles_list):
        data = smiles_to_graph(smiles, max_nodes=max_nodes)
        if data is not None:
            graphs.append(data)
        else:
            skipped += 1

        if show_progress and (i + 1) % 10000 == 0:
            print(
                f"    Processed {i + 1:,}/{len(smiles_list):,} | "
                f"Valid: {len(graphs):,} | Skipped: {skipped:,}"
            )

    return graphs, len(graphs), skipped
