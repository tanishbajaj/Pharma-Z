"""
Pipeline Routes

API endpoints for the full drug discovery pipeline.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.pipeline_schema import (
    PipelineRunRequest,
    PipelineResultResponse,
    PipelineBatchRequest,
    PipelineBatchResponse,
)
from app.services import pipeline_service
from app.utils.logging_config import logger

pipeline_router = APIRouter(prefix="/pipeline")


@pipeline_router.post("/run", response_model=PipelineResultResponse)
async def run_pipeline(request: PipelineRunRequest):
    """
    Run the full drug discovery pipeline:
    validate → binding affinity → toxicity → ranking.
    """
    try:
        result = pipeline_service.run_full_pipeline(
            smiles_input=request.smiles,
            protein_sequence=request.protein_sequence,
            num_generated=request.num_generated,
        )
        return PipelineResultResponse(**result)
    except Exception as error:
        logger.error(f"Pipeline failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@pipeline_router.post("/batch", response_model=PipelineBatchResponse)
async def run_batch_pipeline(request: PipelineBatchRequest):
    """Run pipeline for multiple drug candidates against a single protein target."""
    try:
        ranked = pipeline_service.run_batch_pipeline(
            drug_smiles_list=request.smiles_list,
            protein_sequence=request.protein_sequence,
        )
        return PipelineBatchResponse(candidates=ranked, total_ranked=len(ranked))
    except Exception as error:
        logger.error(f"Batch pipeline failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))
