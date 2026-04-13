"""
Molecule Routes

API endpoints for molecule generation, validation, training, and evaluation.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.molecule_schema import (
    MoleculeGenerateRequest,
    MoleculeGenerateResponse,
    MoleculeValidateRequest,
    MoleculeValidateResponse,
    MoleculeTrainRequest,
    TrainingStatusResponse,
    MoleculeEvaluateRequest,
    EvaluationMetrics,
    GeneratedMolecule,
)
from app.services import molecule_service
from app.utils.logging_config import logger

molecule_router = APIRouter(prefix="/molecules")


# ---------------------------------------------------------------------------
# POST /molecules/generate — Generate novel molecules
# ---------------------------------------------------------------------------

@molecule_router.post("/generate", response_model=MoleculeGenerateResponse)
async def generate_molecules(request: MoleculeGenerateRequest):
    """
    Generate novel molecules by sampling from the trained VAE latent space.

    Requires a trained model (either loaded or trained via /molecules/train).
    """
    try:
        results = molecule_service.generate_molecules(
            num_molecules=request.num_molecules,
            checkpoint_path=request.checkpoint_path,
        )

        molecules = [GeneratedMolecule(**mol) for mol in results]
        valid_count = sum(1 for m in molecules if m.is_valid)

        return MoleculeGenerateResponse(
            molecules=molecules,
            total_generated=len(molecules),
            valid_count=valid_count,
            validity_rate=valid_count / len(molecules) if molecules else 0.0,
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.error(f"Molecule generation failed: {error}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(error)}")


# ---------------------------------------------------------------------------
# POST /molecules/validate — Validate a SMILES string
# ---------------------------------------------------------------------------

@molecule_router.post("/validate", response_model=MoleculeValidateResponse)
async def validate_smiles(request: MoleculeValidateRequest):
    """
    Validate a SMILES string and return its molecular properties if valid.
    """
    result = molecule_service.validate_smiles(request.smiles)
    return MoleculeValidateResponse(**result)


# ---------------------------------------------------------------------------
# POST /molecules/train — Train the SMILES VAE
# ---------------------------------------------------------------------------

@molecule_router.post("/train", response_model=TrainingStatusResponse)
async def train_vae(request: MoleculeTrainRequest):
    """
    Train the SMILES VAE model on a ZINC dataset.

    This is a synchronous endpoint — it blocks until training is complete.
    For large datasets, consider running training via a script instead.
    """
    try:
        # Load dataset
        smiles_list = molecule_service.load_zinc_dataset(
            file_path=request.dataset_path,
            max_samples=request.max_samples,
        )

        # Prepare data loader
        data_loader = molecule_service.prepare_training_data(
            smiles_list=smiles_list,
            batch_size=request.batch_size,
        )

        # Initialize model
        model = molecule_service.initialize_vae_model(latent_dim=request.latent_dim)

        # Train
        training_history = molecule_service.train_smiles_vae(
            data_loader=data_loader,
            model=model,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            kl_weight=request.kl_weight,
        )

        final_loss = training_history["total_loss"][-1] if training_history["total_loss"] else None

        return TrainingStatusResponse(
            status="completed",
            message=f"Training completed: {request.num_epochs} epochs on {len(smiles_list)} samples",
            epochs_completed=request.num_epochs,
            final_loss=final_loss,
            training_history=training_history,
        )

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {str(error)}")
    except Exception as error:
        logger.error(f"Training failed: {error}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(error)}")


# ---------------------------------------------------------------------------
# POST /molecules/evaluate — Evaluate generated molecules
# ---------------------------------------------------------------------------

@molecule_router.post("/evaluate", response_model=EvaluationMetrics)
async def evaluate_molecules(request: MoleculeEvaluateRequest):
    """
    Evaluate a set of generated SMILES strings on validity, novelty, and diversity.
    """
    metrics = molecule_service.evaluate_generated_molecules(
        generated_smiles=request.generated_smiles,
        training_smiles=request.training_smiles,
    )
    return EvaluationMetrics(**metrics)
