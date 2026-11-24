# Pharma-Z

**A Privacy-Preserving, Blockchain-Enabled AI Drug Discovery & Forensic Audit Platform**

Pharma-Z is a research project that integrates AI-based molecular generation, drug-target binding prediction, toxicity screening, blockchain-based forensic auditing, differential privacy, and explainable AI into a single end-to-end drug discovery pipeline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Pharma-Z Pipeline                           │
│                                                                 │
│  ┌──────────┐   ┌──────────┐    ┌──────────┐    ┌────────────┐  │
│  │ Molecule │──▶│ Binding  │──▶│ Toxicity │──▶│  Ranking   │  │
│  │Generator │   │Predictor │    │Predictor │    │  & Audit   │  │
│  │(VAE/GNN) │   │(DeepDTA) │    │  (RF)    │    │(Blockchain)│  │
│  └──────────┘   └──────────┘    └──────────┘    └────────────┘  │
│       ▲                                              │          │
│       │         ┌──────────────────────┐             ▼          │
│       └─────────│  ZINC Dataset (22M)  │    Ranked Drug         │
│                 └──────────────────────┘    Candidates          │
└─────────────────────────────────────────────────────────────────┘
```

## System Modules

| Module | Model | Purpose | Status |
|---|---|---|---|
| **Molecule Generator (Stage 1)** | SMILES VAE (LSTM) | Generate drug-like molecules from SMILES strings | ✅ Implemented |
| **Molecule Generator (Stage 2)** | GraphVAE (GCN) | Generate molecules as molecular graphs | ✅ Implemented |
| **Binding Predictor** | DeepDTA (CNN+LSTM) | Predict drug-protein binding affinity | ✅ Implemented |
| **Toxicity Predictor** | Random Forest | Screen molecules for toxicity | ✅ Implemented |
| **Drug Ranker** | Weighted scoring | Rank candidates by binding, toxicity, drug-likeness | ✅ Implemented |
| **Audit Logger** | SQLite + hashing | Log all predictions with tamper-proof hashes | ✅ Implemented |
| **Privacy Engine** | Opacus (DP-SGD) | Differential privacy for model training | ✅ Implemented |
| **Blockchain Layer** | Solidity + Ganache | Store immutable audit proofs on-chain | ✅ Implemented |
| **Explainability** | SHAP | Explain model decisions for regulatory compliance | ✅ Implemented |
| **Toxicity Predictor** | Masked per-assay RF (Tox21) | Screen molecules against 12 toxicity assays | ✅ Trained — 0.741 scaffold ROC-AUC |
| **Web Interface** | React + Vite | Generate, predict, screen, rank and audit visually | ✅ Implemented |

---

## Dataset: ZINC 2D SMILES

### Source

The training dataset was downloaded from the [ZINC Database](https://zinc.docking.org/) using the **ZINC-downloader-2D-smi.wget** file, which contains 960 download commands pointing to tranch files on `http://files.docking.org/2D/`.

### Download Pipeline

```
ZINC-downloader-2D-smi.wget (960 URLs)
        │
        ▼
download_zinc_wget.py (parallel curl.exe, 10 workers)
        │
        ▼
960 .smi tranch files → ~412 successful downloads
        │
        ▼
Merge & deduplicate → zinc_merged.smi (22,804,768 unique SMILES)
        │
        ▼
prepare_full_dataset.py (RDKit validation + canonicalization)
        │
        ▼
zinc_full.csv     — Full validated dataset (~22M molecules)
zinc_5m.csv       — 5M molecule subset
zinc_1m.csv       — 1M molecule subset
zinc250k.csv      — 250K molecule subset (quick experiments)
```

### How the Download Worked

1. **Parsed** the `.wget` file to extract 960 download URLs from `files.docking.org/2D/`
2. **Downloaded** all tranch `.smi` files in parallel (10 workers) using `curl.exe -sL` (follows HTTP 301 redirects)
3. **Merged** all successful downloads into a single `zinc_merged.smi` (22.8M unique SMILES, 1.16 GB)
4. **Validated** every SMILES string with RDKit, canonicalized, deduplicated, and filtered by length (5–120 chars)
5. **Produced** clean CSVs at multiple scales for training flexibility

### Dataset Statistics

| File | Molecules | Size | Description |
|---|---|---|---|
| `zinc_full.csv` | ~22M | ~1 GB | RDKit-validated, canonicalized, full dataset |
| `zinc_5m.csv` | 5,000,000 | ~200 MB | Medium training set |
| `zinc_1m.csv` | 1,000,000 | ~38 MB | Standard baseline training |
| `zinc250k.csv` | 250,000 | ~8.6 MB | Quick experiments & debugging |
| `zinc_merged.smi` | 22,804,768 | 1,164 MB | Raw merged (pre-validation) |

### Re-downloading

To re-download the dataset from scratch:

```bash
cd backend
python scripts/download_zinc_wget.py --max-molecules 0 --workers 10
python scripts/prepare_full_dataset.py
```

---

## Model Implementations

### 1. SMILES VAE (Stage 1 — Baseline)

**Architecture:** Bidirectional LSTM Encoder → Latent Space (z) → LSTM Decoder

```
Input SMILES → [Embedding] → [BiLSTM Encoder] → μ, σ
                                                   │
                                            reparameterize
                                                   │
                                                   z
                                                   │
                            [LSTM Decoder] ← [z → hidden state]
                                   │
                            Reconstructed SMILES
```

| Component | Details |
|---|---|
| Encoder | Embedding (64d) → BiLSTM (256 hidden, 2 layers) → μ, log_var (128d) |
| Decoder | Latent z (128d) → LSTM (256 hidden, 2 layers) → vocab logits (56 classes) |
| Loss | Cross-Entropy (reconstruction) + β·KL Divergence |
| Vocabulary | 56 tokens (SMILES characters + PAD/START/END) |
| Max length | 120 characters |
| Parameters | ~3.4M |

**Key files:**
- `app/models/smiles_vae.py` — Model (SmilesEncoder, SmilesDecoder, SmilesVAE)
- `app/models/smiles_tokenizer.py` — Character-level tokenizer
- `app/services/molecule_service.py` — Training, generation, evaluation service
- `scripts/train_smiles_vae.py` — Standalone training script

**Training:**
```bash
# Quick test (CPU, ~5 min)
python scripts/train_smiles_vae.py --max-samples 2000 --epochs 5

# Full training (GPU recommended)
python scripts/train_smiles_vae.py --smiles-file data/raw/zinc_1m.csv --epochs 50 --batch-size 256
```

**Evaluation metrics:** Validity, Uniqueness, Novelty, Diversity (Tanimoto)

---

### 2. GraphVAE (Stage 2 — Graph-based)

**Architecture:** GCN Encoder → Latent Space → MLP Decoder (adjacency + node features)

```
Molecular Graph (atoms + bonds)
        │
  [GCN Layer 1] → [GCN Layer 2] → [GCN Layer 3]
        │
  [Global Mean Pooling]
        │
     μ, log_var → reparameterize → z
        │
  [MLP Decoder]
     ├── Adjacency matrix (N×N)
     └── Node features (N×17)
```

| Component | Details |
|---|---|
| Node features | 17-dim: atom type (10 one-hot) + degree (6 one-hot) + aromatic (1) |
| Encoder | 3× GCNConv (17→256→256→256) + Global Mean Pooling → μ, log_var (128d) |
| Decoder | MLP → adjacency matrix (N×N) + node features (N×17) |
| Loss | BCE (adjacency) + MSE (node features) + KL Divergence |
| Max atoms | 38 per molecule |
| Parameters | ~1.5M |

**Key files:**
- `app/models/graph_vae.py` — GraphVAE model with GCN encoder
- `app/models/graph_utils.py` — SMILES ↔ Graph conversion (RDKit)
- `app/models/graph_dataset.py` — PyG InMemoryDataset with disk caching
- `app/services/graph_molecule_service.py` — Training, generation, evaluation
- `scripts/train_graph_vae.py` — Standalone training script

**Training:**
```bash
python scripts/train_graph_vae.py --smiles-file data/raw/zinc_1m.csv --epochs 50 --batch-size 128
```

---

### 3. DeepDTA (Binding Prediction)

Predicts drug-target binding affinity using CNN for drug SMILES and protein sequences.

- **File:** `app/models/deep_dta_model.py`
- **Service:** `app/services/binding_service.py`

### 4. Toxicity Predictor

Random Forest classifier for molecular toxicity screening.

- **File:** `app/models/toxicity_model.py`
- **Service:** `app/services/toxicity_service.py`

### 5. Drug Ranking Pipeline

End-to-end pipeline that generates → predicts binding → screens toxicity → ranks candidates.

- **Service:** `app/services/ranking_service.py`, `app/services/pipeline_service.py`

---

## API Endpoints

All endpoints are served via FastAPI at `http://localhost:8000/api/v1/`.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| **Molecule Generation** | | |
| `/molecules/generate` | POST | Generate molecules (SMILES VAE) |
| `/molecules/validate` | POST | Validate & get properties |
| `/molecules/train` | POST | Train SMILES VAE |
| **Graph Molecule Generation** | | |
| `/molecules/train-graph` | POST | Train GraphVAE |
| `/molecules/generate-graph` | POST | Generate molecules (GraphVAE) |
| `/molecules/evaluate-graph` | POST | Evaluate generation quality |
| **Binding Prediction** | | |
| `/binding/predict` | POST | Predict drug-target binding |
| `/binding/batch-predict` | POST | Batch binding predictions |
| **Toxicity** | | |
| `/toxicity/predict` | POST | Predict toxicity |
| **Pipeline** | | |
| `/pipeline/run` | POST | Full discovery pipeline |
| **Audit** | | |
| `/audit/logs` | GET | Get audit trail |
| **Blockchain** | | |
| `/blockchain/store-hash` | POST | Store audit proof on-chain |
| `/blockchain/verify/{record_id}` | GET | Verify on-chain proof |
| `/blockchain/transactions` | GET | List on-chain audit records |

---

## Project Structure

```
Pharma-Z/
├── README.md
├── .gitignore
└── backend/
    ├── app/
    │   ├── main.py                    # FastAPI application entry point
    │   ├── config.py                  # Settings (paths, defaults)
    │   ├── database.py                # SQLAlchemy database setup
    │   ├── api/routes/
    │   │   ├── molecule_routes.py     # SMILES VAE endpoints
    │   │   ├── graph_molecule_routes.py # GraphVAE endpoints
    │   │   ├── binding_routes.py      # Binding prediction endpoints
    │   │   ├── toxicity_routes.py     # Toxicity screening endpoints
    │   │   ├── pipeline_routes.py     # Full pipeline endpoint
    │   │   ├── audit_routes.py        # Audit log endpoints
    │   │   ├── blockchain_routes.py   # Blockchain audit endpoints
    │   │   └── health_routes.py       # Health check
    │   ├── models/
    │   │   ├── smiles_vae.py          # SMILES VAE (LSTM encoder/decoder)
    │   │   ├── smiles_tokenizer.py    # Character-level SMILES tokenizer
    │   │   ├── graph_vae.py           # GraphVAE (GCN encoder, MLP decoder)
    │   │   ├── graph_utils.py         # SMILES ↔ molecular graph conversion
    │   │   ├── graph_dataset.py       # PyG InMemoryDataset for ZINC
    │   │   ├── deep_dta_model.py      # DeepDTA binding affinity model
    │   │   ├── toxicity_model.py      # Random Forest toxicity classifier
    │   │   └── training_utils.py      # GPU detection, checkpointing
    │   ├── services/
    │   │   ├── molecule_service.py    # SMILES VAE train/generate/evaluate
    │   │   ├── graph_molecule_service.py # GraphVAE train/generate/evaluate
    │   │   ├── binding_service.py     # Binding prediction service
    │   │   ├── toxicity_service.py    # Toxicity prediction service
    │   │   ├── ranking_service.py     # Drug candidate ranking
    │   │   └── pipeline_service.py    # End-to-end pipeline orchestration
    │   ├── schemas/                   # Pydantic request/response schemas
    │   ├── audit/                     # Hash-based forensic audit logging
    │   ├── privacy/                   # Differential privacy (DP-SGD)
    │   ├── blockchain/                # Solidity smart contract integration
    │   ├── explainability/            # SHAP-based model explanations
    │   └── utils/
    │       ├── constants.py           # SMILES charset, model defaults
    │       └── logging_config.py      # Structured logging
    ├── data/
    │   ├── ZINC-downloader-2D-smi.wget  # Source wget file (960 tranch URLs)
    │   ├── raw/                         # Downloaded & merged SMILES data
    │   ├── processed/                   # Cached graph datasets
    │   └── saved_models/                # Training checkpoints
    ├── scripts/
    │   ├── download_zinc_wget.py      # Download ZINC 2D tranches
    │   ├── prepare_full_dataset.py    # RDKit validate & create CSVs
    │   ├── train_smiles_vae.py        # Phase 2: SMILES VAE training
    │   └── train_graph_vae.py         # Phase 3: GraphVAE training
    ├── tests/                         # Unit & integration tests
    ├── logs/                          # Application logs
    ├── requirements.txt               # Python dependencies
    ├── pyproject.toml                 # Project metadata
    └── .env                           # Environment configuration
```

---

## Quick Start

### 1. Setup

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
# For GPU training:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cp .env.example .env
```

### 2. Download Dataset (if not already present)

```bash
python scripts/download_zinc_wget.py --max-molecules 0 --workers 10
python scripts/prepare_full_dataset.py
```

### 3. Train Models

```bash
# Phase 2: SMILES VAE baseline
python scripts/train_smiles_vae.py --smiles-file data/raw/zinc_1m.csv --epochs 50

# Phase 3: GraphVAE
python scripts/train_graph_vae.py --smiles-file data/raw/zinc_1m.csv --epochs 50
```

### 4. Build the Frontend

```bash
cd frontend
npm install
npm run build
```

This produces `frontend/dist/`, which the FastAPI app serves at `/`. Fonts and
the 2D structure renderer are bundled locally, so the interface works with no
internet connection.

For frontend development with hot reload (API calls proxy to port 8000):

```bash
npm run dev     # http://localhost:5173
```

### 5. Run API Server

```bash
cd backend
uvicorn app.main:app --reload
```

- [http://localhost:8000](http://localhost:8000) — the Pharma-Z interface
- [http://localhost:8000/docs](http://localhost:8000/docs) — Swagger API

### 6. Blockchain Audit Trail (optional)

The ledger view needs a local chain. In a separate terminal:

```bash
npx ganache --port 8545 --deterministic
```

Then compile, deploy, and verify end to end:

```bash
cd backend
python scripts/deploy_contract.py        # writes CONTRACT_ADDRESS into .env
python scripts/test_blockchain_e2e.py    # 29 checks incl. tamper detection
```

Ganache is in-memory, so a restart wipes the chain and requires redeploying.
Add `--db ./ganache_db` to persist it between runs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI + Uvicorn |
| **ML / Deep Learning** | PyTorch, PyTorch Geometric |
| **Chemistry** | RDKit (SMILES parsing, molecular properties, graph conversion) |
| **Classical ML** | scikit-learn (Random Forest) |
| **Blockchain** | Solidity + Ganache + Web3.py |
| **Privacy** | Opacus (DP-SGD) |
| **Explainability** | SHAP |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **Experiment Tracking** | MLflow |
| **Testing** | pytest + httpx |

---

## Training Phases

| Phase | Model | Dataset | Status |
|---|---|---|---|
| Phase 1 | Data pipeline | ZINC 22M download + validation | ✅ Complete |
| Phase 2 | SMILES VAE | ZINC 1M (baseline) | ✅ Pipeline validated |
| Phase 3 | GraphVAE | ZINC 1M | ⏳ Ready for GPU training |
| Phase 4 | DeepDTA | KIBA dataset | ⏳ Pending |
| Phase 5 | Full pipeline | Integration | ⏳ Pending |

---

## License

Research project — not for commercial use.
