"""
Binding Affinity Routes

API endpoints for drug-protein binding affinity prediction using DeepDTA.
Supports both baseline (CNN) and ESM-enhanced models.

Endpoints:
    POST /binding/predict       — Single drug-protein prediction
    POST /binding/batch-predict — Batch prediction
    POST /binding/train         — Train DeepDTA model
    GET  /binding/model-info    — Get loaded model info
"""

import os
import time

from fastapi import APIRouter, HTTPException

from app.schemas.binding_schema import (
    BindingPredictRequest,
    BindingPredictResponse,
    BindingBatchPredictRequest,
    BindingBatchPredictResponse,
    BindingTrainRequest,
    BindingTrainResponse,
    BindingTrainMetrics,
)
from app.services import binding_service
from app.config import settings
from app.utils.logging_config import logger

binding_router = APIRouter(prefix="/binding")


# ---------------------------------------------------------------------------
# POST /binding/predict — Predict binding affinity for a drug-protein pair
# ---------------------------------------------------------------------------

@binding_router.post("/predict", response_model=BindingPredictResponse)
async def predict_binding(request: BindingPredictRequest):
    """
    Predict binding affinity score for a single drug-protein pair.

    Requires a trained DeepDTA model. If no model is loaded, it will
    auto-load from the default checkpoint path.
    """
    try:
        # Auto-load best checkpoint if no model is loaded
        checkpoint = request.checkpoint_path
        if checkpoint is None:
            checkpoint = _find_best_checkpoint()

        result = binding_service.predict_binding_affinity(
            drug_smiles=request.drug_smiles,
            protein_sequence=request.protein_sequence,
            checkpoint_path=checkpoint,
        )
        return BindingPredictResponse(**result)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found. Train first via POST /binding/train or scripts/train_deep_dta.py. Details: {error}",
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.error(f"Binding prediction failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


# ---------------------------------------------------------------------------
# POST /binding/batch-predict — Batch binding affinity prediction
# ---------------------------------------------------------------------------

@binding_router.post("/batch-predict", response_model=BindingBatchPredictResponse)
async def predict_binding_batch(request: BindingBatchPredictRequest):
    """
    Predict binding affinity for multiple drug-protein pairs.

    Accepts lists of drug SMILES and protein sequences (must be same length).
    """
    try:
        if len(request.drug_smiles_list) != len(request.protein_sequence_list):
            raise ValueError(
                f"Mismatched list lengths: drugs={len(request.drug_smiles_list)}, "
                f"proteins={len(request.protein_sequence_list)}"
            )

        results = binding_service.predict_binding_batch(
            drug_smiles_list=request.drug_smiles_list,
            protein_sequence_list=request.protein_sequence_list,
        )
        predictions = [BindingPredictResponse(**r) for r in results]
        return BindingBatchPredictResponse(
            predictions=predictions,
            total_predictions=len(predictions),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.error(f"Batch binding prediction failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


# ---------------------------------------------------------------------------
# POST /binding/train — Train DeepDTA model
# ---------------------------------------------------------------------------

@binding_router.post("/train", response_model=BindingTrainResponse)
async def train_model(request: BindingTrainRequest):
    """
    Train the DeepDTA binding affinity model.

    Supports both baseline (CNN) and ESM-enhanced model variants.
    This is a synchronous endpoint — it blocks until training is complete.
    For full training, use scripts/train_deep_dta.py instead.
    """
    try:
        start_time = time.time()

        # Load dataset
        dataframe = binding_service.load_binding_dataset(
            dataset_name=request.dataset_name,
            file_path=request.dataset_path,
            max_samples=request.max_samples,
        )

        # Split data
        train_df, val_df, test_df = binding_service.create_train_val_test_split(
            dataframe,
            split_mode=request.split_mode,
        )

        # Prepare data loaders
        train_loader = binding_service.prepare_binding_data(
            drug_smiles_list=train_df["smiles"].tolist(),
            protein_sequence_list=train_df["protein_sequence"].tolist(),
            affinity_scores=train_df["affinity_score"].tolist(),
            batch_size=request.batch_size,
            shuffle=True,
        )

        val_loader = binding_service.prepare_binding_data(
            drug_smiles_list=val_df["smiles"].tolist(),
            protein_sequence_list=val_df["protein_sequence"].tolist(),
            affinity_scores=val_df["affinity_score"].tolist(),
            batch_size=request.batch_size,
            shuffle=False,
        )

        # Initialize model
        model = binding_service.initialize_dta_model(
            model_type=request.model_type,
        )

        # Train with validation
        checkpoint_dir = os.path.join(
            settings.saved_models_dir,
            f"deep_dta{'_esm' if request.model_type == 'esm' else ''}",
        )

        history = binding_service.train_deep_dta(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            checkpoint_dir=checkpoint_dir,
            patience=request.patience,
        )

        # Evaluate on test set
        test_loader = binding_service.prepare_binding_data(
            drug_smiles_list=test_df["smiles"].tolist(),
            protein_sequence_list=test_df["protein_sequence"].tolist(),
            affinity_scores=test_df["affinity_score"].tolist(),
            batch_size=request.batch_size,
            shuffle=False,
        )

        device = next(model.parameters()).device
        test_metrics_raw = binding_service.evaluate_model(model, test_loader, device)

        test_metrics = BindingTrainMetrics(
            mse=test_metrics_raw["mse"],
            rmse=test_metrics_raw["rmse"],
            concordance_index=test_metrics_raw["ci"],
            pearson_r=test_metrics_raw["pearson_r"],
        )

        training_time = time.time() - start_time
        best_ci = max(history.get("val_ci", [0.0]))
        best_checkpoint = os.path.join(checkpoint_dir, "best_model.pt")

        return BindingTrainResponse(
            status="success",
            model_type=request.model_type,
            dataset=request.dataset_name,
            epochs_completed=len(history.get("train_mse", [])),
            best_val_ci=best_ci,
            test_metrics=test_metrics,
            training_time_seconds=round(training_time, 2),
            checkpoint_path=best_checkpoint,
        )

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {error}")
    except Exception as error:
        logger.error(f"Binding training failed: {error}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(error)}")


# ---------------------------------------------------------------------------
# GET /binding/model-info — Get loaded model information
# ---------------------------------------------------------------------------

@binding_router.get("/model-info")
async def get_model_info():
    """
    Get information about available trained models.

    Returns which checkpoints exist and their metadata.
    """
    models = []

    for model_type, subdir in [("baseline", "deep_dta"), ("esm", "deep_dta_esm")]:
        checkpoint_dir = os.path.join(settings.saved_models_dir, subdir)
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        results_path = os.path.join(checkpoint_dir, "training_results.json")

        if os.path.exists(best_model_path):
            size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
            info = {
                "model_type": model_type,
                "checkpoint_path": best_model_path,
                "checkpoint_size_mb": round(size_mb, 2),
                "ready": True,
            }

            # Load training results if available
            if os.path.exists(results_path):
                import json
                with open(results_path) as f:
                    results = json.load(f)
                info["test_metrics"] = results.get("test_metrics", {})
                info["training_time_minutes"] = results.get("training_time_minutes")
                info["dataset"] = results.get("dataset")

            models.append(info)
        else:
            models.append({
                "model_type": model_type,
                "checkpoint_path": best_model_path,
                "ready": False,
                "message": "Not trained yet. Use POST /binding/train or scripts/train_deep_dta.py",
            })

    return {"models": models}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_best_checkpoint() -> str:
    """Find the best available checkpoint (prefer ESM over baseline)."""
    # Prefer ESM model if available
    esm_path = os.path.join(settings.saved_models_dir, "deep_dta_esm", "best_model.pt")
    if os.path.exists(esm_path):
        return esm_path

    baseline_path = os.path.join(settings.saved_models_dir, "deep_dta", "best_model.pt")
    if os.path.exists(baseline_path):
        return baseline_path

    raise FileNotFoundError("No trained binding affinity model found.")
