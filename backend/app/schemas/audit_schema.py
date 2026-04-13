"""
Audit Schemas

Pydantic models for audit log endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AuditLogEntry(BaseModel):
    """A single audit log entry."""
    log_id: str
    timestamp: str
    event_type: str
    data: Dict[str, Any] = {}
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    dataset_name: Optional[str] = None
    user_id: Optional[str] = None
    data_hash: str


class AuditLogListResponse(BaseModel):
    """Response for listing audit logs."""
    logs: List[AuditLogEntry]
    total: int


class AuditVerifyResponse(BaseModel):
    """Response for audit log integrity verification."""
    log_id: str
    is_valid: bool
    original_hash: Optional[str] = None
    computed_hash: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None


class AuditLineageResponse(BaseModel):
    """Response for data lineage trace."""
    entity_id: str
    lineage: List[AuditLogEntry]
    total_events: int
