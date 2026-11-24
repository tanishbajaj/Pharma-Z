# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite (user decision). No frontend or build tooling existed before this;
the backend is FastAPI + Python. The built `dist/` is served by the existing
FastAPI app so the whole system runs from one `uvicorn` command.

## Users

Two audiences, both present at the same moment:

- **The project team** (three B.Tech Electronics & Communication students at
  LNMIIT Jaipur) driving a live demonstration.
- **Academic examiners** in a viva, who have read the report and will probe
  whether the claimed results are real. They are domain-literate in
  engineering but not necessarily in cheminformatics.

The job is not day-to-day research use. It is: make a multi-model ML pipeline
legible and believable to someone watching it run for the first time, in a
single sitting, without the operator typing JSON into Swagger.

## Product Purpose

Pharma-Z chains four ML stages into one drug-discovery pipeline — generate
candidate molecules, predict drug–target binding affinity, screen for
toxicity, rank the survivors — and writes a tamper-proof hash of every
decision to an Ethereum audit trail.

Success for this surface is that an examiner watches a molecule travel the
full pipeline and can see, at each stage, what the model decided and why that
decision is defensible.

## Positioning

Existing tools cover one stage each: REINVENT generates, AutoDock docks,
DeepChem predicts properties. Commercial end-to-end platforms (Atomwise,
Insilico) are closed source. Pharma-Z's claim is the *seam*: generation,
binding, toxicity and ranking in one pipeline, with blockchain audit,
differential privacy and SHAP explainability attached to it.

The audit trail is the genuinely unusual part — no comparable open platform
records pipeline decisions as immutable on-chain hashes.

## Operating Context

- Demonstrated on a laptop or over a projector, usually with the backend on
  `localhost:8000` and Ganache on `localhost:8545`.
- Model inference is **not instant**: molecule generation and batch binding
  prediction take seconds to tens of seconds. Waiting is part of the real
  usage scene and must be designed for, not hidden.
- Network at demo time is not guaranteed. Anything fetched from a CDN is a
  risk the design must account for.
- GPU training happens separately on an institutional DGX cluster; the UI
  never trains, it only runs inference against saved checkpoints.

## Capabilities and Constraints

Working REST API, all under `/api/v1`:

| Area | Endpoints |
|---|---|
| Molecules (SMILES VAE) | `POST /molecules/generate`, `POST /molecules/validate` |
| Molecules (GraphVAE) | `POST /molecules/generate-graph`, `POST /molecules/evaluate-graph` |
| Binding | `POST /binding/predict`, `POST /binding/batch-predict`, `GET /binding/model-info` |
| Toxicity | `POST /toxicity/predict` |
| Pipeline | `POST /pipeline/run` |
| Audit | `GET /audit/logs` |
| Blockchain | `POST /blockchain/store-hash`, `GET /blockchain/verify/{id}`, `GET /blockchain/transactions` |
| Health | `GET /health` |

Constraints:

- Molecules are exchanged as SMILES strings. 2D structure rendering is a
  **frontend** concern — the backend returns no images or coordinates.
- Toxicity returns 12 per-assay probabilities plus one aggregated score; the
  gate is `noisy_or > 0.67`, chosen on a validation split.
- Binding affinity is a KIBA score, clamped to [0, 17.2]. Higher is stronger.
- `/pipeline/run` needs both a SMILES string and a protein amino-acid
  sequence; protein sequences are long (GSK3B is ~420 residues).
- CORS is already open (`allow_origins=["*"]`) in `backend/app/main.py`.

Undecided: no authentication, no multi-user accounts, no persistence of user
sessions. Not required for this project.

## Brand Commitments

- Name: **Pharma-Z**. Full title: "A Privacy-Preserving, Blockchain-Enabled
  AI Drug Discovery & Forensic Audit Platform".
- No existing logo, typeface, or palette. Nothing is binding.
- Terminology used throughout the report and code, which the UI should match:
  SMILES, binding affinity, KIBA score, Concordance Index (CI), QED, LogP,
  TPSA, validity / uniqueness / novelty, Tox21 assays.

## Evidence on Hand

Real, measured results that the UI may display as fact:

- SMILES VAE: 91.6% validity, 100% uniqueness, 99.7% novelty (10,000 samples)
- GraphVAE V2: 52.6% validity, 99.1% uniqueness, 100% novelty (2,000 samples)
- DeepDTA baseline: CI 0.846, MSE 2.461 on KIBA held-out test (11,825 pairs)
- DeepDTA + ESM-2: CI 0.842, MSE 1.528
- Toxicity RF: mean ROC-AUC 0.7411 ± 0.0016 (scaffold split, 5 seeds);
  0.8270 ± 0.0098 (random split)
- Blockchain: AuditTrail.sol deployed to Ganache, 29/29 end-to-end checks pass

Absences that must not be fabricated:

- **No real users, no testimonials, no customers, no press.**
- No wet-lab validation. Nothing here has been tested on an actual compound.
- Not deployed anywhere public; there is no hosted instance or uptime claim.
- No pricing, licensing, or commercial status. The repo states "Research
  project — not for commercial use."

## Product Principles

1. **Show the pipeline as a pipeline.** The product's claim is the seam
   between stages, so the interface must make hand-offs visible rather than
   presenting four unrelated tools.
2. **Numbers carry their provenance.** Every metric shown should be traceable
   to which model, which dataset, and which split produced it. The project's
   credibility rests on not overclaiming.
3. **Latency is content.** Inference takes real time; show what is running and
   on how many molecules rather than an undifferentiated spinner.
4. **Survive a bad network.** The demo may run offline. Nothing essential may
   depend on a request to the public internet at the moment of truth.
5. **Chemistry stays legible.** A SMILES string is not evidence to a viewer;
   a drawn structure with its properties is.

## Accessibility & Inclusion

No formal standard was established for this project. The surface will be
operated live on a projector, which makes contrast and type size at distance
a practical requirement rather than a compliance one.
