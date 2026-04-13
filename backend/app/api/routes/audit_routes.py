"""
Audit Routes

API endpoints for forensic audit log access and verification.
"""

from fastapi import APIRouter, HTTPException, Query

from app.schemas.audit_schema import (
    AuditLogEntry,
    AuditLogListResponse,
    AuditVerifyResponse,
    AuditLineageResponse,
)
from app.audit import audit_logger

audit_router = APIRouter(prefix="/audit")


@audit_router.get("/logs", response_model=AuditLogListResponse)
async def list_audit_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    event_type: str = Query(default=None),
):
    """List audit log entries with optional filtering."""
    logs = audit_logger.get_all_logs(limit=limit, offset=offset, event_type=event_type)
    return AuditLogListResponse(logs=logs, total=len(logs))


@audit_router.get("/logs/{log_id}", response_model=AuditLogEntry)
async def get_audit_log(log_id: str):
    """Get a single audit log entry by ID."""
    entry = audit_logger.get_log_by_id(log_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Audit log {log_id} not found")
    return AuditLogEntry(**entry)


@audit_router.get("/logs/{log_id}/verify", response_model=AuditVerifyResponse)
async def verify_audit_log(log_id: str):
    """Verify the integrity of an audit log entry by recomputing its hash."""
    result = audit_logger.verify_log_integrity(log_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return AuditVerifyResponse(**result)


@audit_router.get("/lineage/{entity_id}", response_model=AuditLineageResponse)
async def trace_lineage(entity_id: str):
    """Trace the full data lineage of an entity across all audit logs."""
    lineage = audit_logger.get_lineage(entity_id)
    return AuditLineageResponse(
        entity_id=entity_id,
        lineage=lineage,
        total_events=len(lineage),
    )
