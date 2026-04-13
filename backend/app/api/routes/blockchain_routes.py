"""
Blockchain Routes

API endpoints for blockchain hash storage and verification.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.blockchain import contract_manager
from app.utils.logging_config import logger

blockchain_router = APIRouter(prefix="/blockchain")


# --- Schemas ---

class StoreHashRequest(BaseModel):
    """Request to store a hash on the blockchain."""
    data_hash: str = Field(..., description="SHA-256 hash to store")
    event_type: str = Field(default="prediction", description="Type of audit event")


class StoreHashResponse(BaseModel):
    """Response from storing a hash."""
    status: str
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    record_id: Optional[int] = None
    data_hash: Optional[str] = None
    event_type: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


class VerifyHashResponse(BaseModel):
    """Response from verifying a hash."""
    status: str
    record_id: Optional[int] = None
    is_valid: bool = False
    stored_hash: Optional[str] = None
    provided_hash: Optional[str] = None
    event_type: Optional[str] = None
    block_timestamp: Optional[int] = None
    error: Optional[str] = None


# --- Routes ---

@blockchain_router.post("/store-hash", response_model=StoreHashResponse)
async def store_hash(request: StoreHashRequest):
    """Store a data hash on the blockchain for tamper-proof auditing."""
    result = contract_manager.store_hash_on_chain(
        data_hash=request.data_hash,
        event_type=request.event_type,
    )
    return StoreHashResponse(**result)


@blockchain_router.get("/verify/{record_id}", response_model=VerifyHashResponse)
async def verify_hash(record_id: int, data_hash: str = Query(...)):
    """Verify a hash against the blockchain record."""
    result = contract_manager.verify_hash_on_chain(
        record_id=record_id,
        data_hash=data_hash,
    )
    return VerifyHashResponse(**result)


@blockchain_router.get("/transactions")
async def list_transactions(limit: int = Query(default=100, ge=1, le=1000)):
    """List recent blockchain audit transactions."""
    transactions = contract_manager.get_all_transactions(limit=limit)
    return {"transactions": transactions, "total": len(transactions)}
