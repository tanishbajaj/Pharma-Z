"""
Pharma-Z Constants

Application-wide constants used across all modules.
"""

# ---------------------------------------------------------------------------
# SMILES Tokenizer
# ---------------------------------------------------------------------------

SMILES_CHARSET = [
    " ", "#", "%", "(", ")", "+", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "=", "@",
    "A", "B", "C", "F", "H", "I", "K", "L", "M", "N", "O", "P", "S",
    "V", "Z",
    "[", "\\", "]",
    "a", "b", "c", "e", "g", "i", "l", "n", "o", "p", "r", "s", "t", "u",
]

PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"

MAX_MOLECULE_LENGTH = 120  # Max SMILES string length

# ---------------------------------------------------------------------------
# Model Defaults
# ---------------------------------------------------------------------------

DEFAULT_LATENT_DIM = 128
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_DIM = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 3e-4  # MOSES benchmark (Polykovskiy et al., 2020)

# ---------------------------------------------------------------------------
# Ranking Weights
# ---------------------------------------------------------------------------

RANKING_WEIGHT_BINDING = 0.5    # w1
RANKING_WEIGHT_TOXICITY = 0.3   # w2
RANKING_WEIGHT_DRUGLIKENESS = 0.2  # w3

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------

ZINC250K_FILENAME = "zinc250k.csv"
KIBA_FILENAME = "kiba.csv"
DAVIS_FILENAME = "davis.csv"
TOX21_FILENAME = "tox21.csv"

# ---------------------------------------------------------------------------
# Binding Affinity Model (DeepDTA paper-accurate hyperparameters)
# These are SEPARATE from the VAE defaults above to avoid conflicts.
# ---------------------------------------------------------------------------

BINDING_EMBEDDING_DIM = 128       # Paper uses 128-dim embeddings (not 64)
BINDING_DEFAULT_BATCH_SIZE = 256  # Paper Table 2
BINDING_DEFAULT_EPOCHS = 100      # Paper Table 2
BINDING_DEFAULT_LR = 0.001        # Paper: Adam with lr=0.001

# ---------------------------------------------------------------------------
# Blockchain
# ---------------------------------------------------------------------------

GANACHE_DEFAULT_URL = "http://127.0.0.1:8545"
