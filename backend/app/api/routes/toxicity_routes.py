"""
Toxicity Routes

API endpoints for toxicity prediction.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.toxicity_schema import (
    ToxicityPredictRequest,
    ToxicityPredictResponse,
    ToxicityBatchPredictRequest,
    ToxicityBatchPredictResponse,
    ToxicityTrainRequest,
)
from app.services import toxicity_service
from app.utils.logging_config import logger

toxicity_router = APIRouter(prefix="/toxicity")


@toxicity_router.post("/predict", response_model=ToxicityPredictResponse)
async def predict_toxicity(request: ToxicityPredictRequest):
    """Predict toxicity of a single molecule."""
    try:
        result = toxicity_service.predict_toxicity(
            smiles=request.smiles,
            model_path=request.model_path,
        )
        return ToxicityPredictResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.error(f"Toxicity prediction failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@toxicity_router.post("/batch-predict", response_model=ToxicityBatchPredictResponse)
async def predict_toxicity_batch(request: ToxicityBatchPredictRequest):
    """Predict toxicity for multiple molecules."""
    try:
        results = toxicity_service.predict_toxicity_batch(request.smiles_list)
        predictions = [ToxicityPredictResponse(**r) for r in results]
        toxic_count = sum(1 for p in predictions if p.is_toxic)
        return ToxicityBatchPredictResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            toxic_count=toxic_count,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.error(f"Batch toxicity prediction failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@toxicity_router.post("/train")
async def train_model(request: ToxicityTrainRequest):
    """Train the toxicity prediction model on Tox21 data."""
    try:
        result = toxicity_service.train_toxicity_model(
            dataset_path=request.dataset_path,
            max_samples=request.max_samples,
        )
        return result
    except Exception as error:
        logger.error(f"Toxicity training failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))
