"""
Graph Molecule Routes — API endpoints for GraphVAE (Stage 2)

Endpoints:
    POST /molecules/train-graph      — Train the GraphVAE model
    POST /molecules/generate-graph   — Generate molecules from latent space
    POST /molecules/evaluate-graph   — Evaluate generation quality
"""

import time

from fastapi import APIRouter, HTTPException

from app.schemas.graph_molecule_schema import (
    GraphTrainRequest,
    GraphTrainResponse,
    GraphGenerateRequest,
    GraphGenerateResponse,
    GeneratedMolecule,
    GraphEvalRequest,
    GraphEvalResponse,
)
from app.services.graph_molecule_service import (
    load_graph_dataset,
    create_graph_dataloader,
    initialize_graph_vae,
    train_graph_vae,
    generate_molecules_graph,
    evaluate_graph_generation,
)


graph_molecule_router = APIRouter()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

@graph_molecule_router.post(
    "/molecules/train-graph",
    response_model=GraphTrainResponse,
    summary="Train the GraphVAE model",
)
async def train_graph_model(request: GraphTrainRequest):
    """
    Train the Graph VAE on ZINC SMILES data.

    This is a long-running operation. For large datasets,
    use the standalone script: `python scripts/train_graph_vae.py`
    """
    try:
        t0 = time.time()

        # Load dataset
        dataset = load_graph_dataset(
            smiles_file=request.smiles_file,
            max_nodes=request.max_nodes,
            max_samples=request.max_samples,
        )

        # Create data loader
        loader = create_graph_dataloader(
            dataset,
            batch_size=request.batch_size,
        )

        # Initialize model
        model = initialize_graph_vae(
            max_nodes=request.max_nodes,
            latent_dim=request.latent_dim,
            hidden_dim=request.hidden_dim,
        )

        # Train
        history = train_graph_vae(
            data_loader=loader,
            model=model,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            kl_weight=request.kl_weight,
        )

        training_time = time.time() - t0

        return GraphTrainResponse(
            status="completed",
            dataset_size=len(dataset),
            num_epochs=request.num_epochs,
            final_loss=history["total_loss"][-1],
            final_adj_loss=history["recon_adj_loss"][-1],
            final_node_loss=history["recon_node_loss"][-1],
            final_kl_loss=history["kl_loss"][-1],
            training_time_seconds=round(training_time, 1),
            checkpoint_dir="data/saved_models/graph_vae/",
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

@graph_molecule_router.post(
    "/molecules/generate-graph",
    response_model=GraphGenerateResponse,
    summary="Generate molecules via GraphVAE",
)
async def generate_graph_molecules(request: GraphGenerateRequest):
    """Generate novel molecules by sampling from the GraphVAE latent space."""
    try:
        results = generate_molecules_graph(
            num_molecules=request.num_molecules,
            checkpoint_path=request.checkpoint_path,
            temperature=request.temperature,
            adj_threshold=request.adj_threshold,
        )

        valid_count = sum(1 for r in results if r["is_valid"])

        return GraphGenerateResponse(
            num_generated=len(results),
            num_valid=valid_count,
            validity_rate=round(
                valid_count / len(results) if results else 0, 4
            ),
            molecules=[
                GeneratedMolecule(**r) for r in results
            ],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@graph_molecule_router.post(
    "/molecules/evaluate-graph",
    response_model=GraphEvalResponse,
    summary="Evaluate GraphVAE generation quality",
)
async def evaluate_graph_model(request: GraphEvalRequest):
    """Generate molecules and compute validity/uniqueness/novelty metrics."""
    try:
        metrics = evaluate_graph_generation(
            num_samples=request.num_samples,
            temperature=request.temperature,
        )

        return GraphEvalResponse(**metrics)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
