"""
Binding Affinity Routes

API endpoints for drug-protein binding affinity prediction.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.binding_schema import (
    BindingPredictRequest,
    BindingPredictResponse,
    BindingBatchPredictRequest,
    BindingBatchPredictResponse,
    BindingTrainRequest,
)
from app.services import binding_service
from app.utils.logging_config import logger

binding_router = APIRouter(prefix="/binding")


@binding_router.post("/predict", response_model=BindingPredictResponse)
async def predict_binding(request: BindingPredictRequest):
    """Predict binding affinity for a drug-protein pair."""
    try:
        result = binding_service.predict_binding_affinity(
            drug_smiles=request.drug_smiles,
            protein_sequence=request.protein_sequence,
            checkpoint_path=request.checkpoint_path,
        )
        return BindingPredictResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.error(f"Binding prediction failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@binding_router.post("/batch-predict", response_model=BindingBatchPredictResponse)
async def predict_binding_batch(request: BindingBatchPredictRequest):
    """Predict binding affinity for multiple drug-protein pairs."""
    try:
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


@binding_router.post("/train")
async def train_model(request: BindingTrainRequest):
    """Train the DeepDTA binding affinity model."""
    try:
        dataframe = binding_service.load_kiba_dataset(
            file_path=request.dataset_path,
            max_samples=request.max_samples,
        )
        data_loader = binding_service.prepare_binding_data(
            drug_smiles_list=dataframe["smiles"].tolist(),
            protein_sequence_list=dataframe["protein_sequence"].tolist(),
            affinity_scores=dataframe["affinity_score"].tolist(),
            batch_size=request.batch_size,
        )
        history = binding_service.train_deep_dta(
            data_loader=data_loader,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
        )
        return {
            "status": "completed",
            "message": f"DeepDTA trained for {request.num_epochs} epochs",
            "training_history": history,
        }
    except Exception as error:
        logger.error(f"Binding training failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))
