"""
Hash Generator

SHA-256 hashing utility for creating tamper-proof data fingerprints.
Used by the audit and blockchain layers.
"""

import hashlib
import json
from typing import Any


def hash_data(data: Any) -> str:
    """
    Generate a SHA-256 hash of any data structure.

    Serializes the data to a deterministic JSON string (sorted keys)
    and returns its hex digest.

    Args:
        data: Any JSON-serializable data (dict, list, str, etc.).

    Returns:
        64-character hex string (SHA-256 hash).
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_string(text: str) -> str:
    """Generate a SHA-256 hash of a plain string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_hash(data: Any, expected_hash: str) -> bool:
    """
    Verify that data matches an expected hash.

    Args:
        data:          The original data.
        expected_hash: The expected SHA-256 hex digest.

    Returns:
        True if hashes match, False otherwise.
    """
    computed_hash = hash_data(data)
    return computed_hash == expected_hash
