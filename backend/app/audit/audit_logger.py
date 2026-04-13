"""
Audit Logger

Forensic audit system for tracking all predictions, model decisions,
and data operations. Writes to both JSON log file and database.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, List

from app.config import settings
from app.audit.hash_generator import hash_data
from app.utils.logging_config import logger


# ---------------------------------------------------------------------------
# In-memory audit store (backed by JSON file)
# ---------------------------------------------------------------------------

_audit_log_file: Optional[str] = None
_audit_entries: List[dict] = []


def _get_log_file_path() -> str:
    """Get or create the audit log file path."""
    global _audit_log_file
    if _audit_log_file is None:
        os.makedirs(settings.logs_dir, exist_ok=True)
        _audit_log_file = os.path.join(settings.logs_dir, "audit.json")
    return _audit_log_file


def _load_existing_entries():
    """Load existing audit entries from the JSON file."""
    global _audit_entries
    log_path = _get_log_file_path()
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                content = f.read().strip()
                if content:
                    _audit_entries = [
                        json.loads(line) for line in content.split("\n") if line.strip()
                    ]
        except Exception as error:
            logger.warning(f"Failed to load existing audit log: {error}")
            _audit_entries = []


# ---------------------------------------------------------------------------
# Core Logging Function
# ---------------------------------------------------------------------------

def log_event(
    event_type: str,
    data: dict,
    model_name: str = None,
    model_version: str = None,
    dataset_name: str = None,
    user_id: str = None,
) -> dict:
    """
    Log an auditable event.

    Args:
        event_type:    Type of event (e.g., "prediction", "training", "data_load").
        data:          Event data payload.
        model_name:    Name of the ML model involved.
        model_version: Version of the model.
        dataset_name:  Dataset used.
        user_id:       User who triggered the event.

    Returns:
        The created audit log entry dict.
    """
    entry = {
        "log_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "data": data,
        "model_name": model_name,
        "model_version": model_version,
        "dataset_name": dataset_name,
        "user_id": user_id,
        "data_hash": hash_data(data),
    }

    # Append to in-memory list
    _audit_entries.append(entry)

    # Append to JSON file
    log_path = _get_log_file_path()
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as error:
        logger.error(f"Failed to write audit log: {error}")

    logger.info(f"Audit logged: [{event_type}] id={entry['log_id'][:8]}...")
    return entry


# ---------------------------------------------------------------------------
# Query Functions
# ---------------------------------------------------------------------------

def get_all_logs(
    limit: int = 100,
    offset: int = 0,
    event_type: str = None,
) -> List[dict]:
    """
    Retrieve audit log entries.

    Args:
        limit:      Maximum entries to return.
        offset:     Number of entries to skip.
        event_type: Filter by event type.

    Returns:
        List of audit log entry dicts.
    """
    if not _audit_entries:
        _load_existing_entries()

    entries = _audit_entries
    if event_type:
        entries = [e for e in entries if e.get("event_type") == event_type]

    # Return in reverse chronological order
    entries = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries[offset: offset + limit]


def get_log_by_id(log_id: str) -> Optional[dict]:
    """Retrieve a single audit log by its ID."""
    if not _audit_entries:
        _load_existing_entries()

    for entry in _audit_entries:
        if entry.get("log_id") == log_id:
            return entry
    return None


def verify_log_integrity(log_id: str) -> dict:
    """
    Verify the integrity of an audit log entry by recomputing its hash.

    Returns:
        Dict with log_id, is_valid, original_hash, computed_hash.
    """
    entry = get_log_by_id(log_id)
    if entry is None:
        return {"log_id": log_id, "is_valid": False, "error": "Log entry not found"}

    original_hash = entry.get("data_hash", "")
    computed_hash = hash_data(entry.get("data", {}))

    return {
        "log_id": log_id,
        "is_valid": original_hash == computed_hash,
        "original_hash": original_hash,
        "computed_hash": computed_hash,
        "timestamp": entry.get("timestamp"),
    }


def get_lineage(entity_id: str) -> List[dict]:
    """
    Trace the full data lineage of an entity (e.g., a molecule or prediction).

    Searches all logs for references to the given entity_id.
    """
    if not _audit_entries:
        _load_existing_entries()

    lineage = []
    for entry in _audit_entries:
        data = entry.get("data", {})
        data_str = json.dumps(data, default=str)
        if entity_id in data_str:
            lineage.append(entry)

    return sorted(lineage, key=lambda e: e.get("timestamp", ""))
