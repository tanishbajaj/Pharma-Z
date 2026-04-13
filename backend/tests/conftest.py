"""
Test Configuration & Shared Fixtures

Used by all test files via pytest's automatic fixture discovery.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def api_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_smiles_list():
    """Return a small list of valid SMILES strings for testing."""
    return [
        "CCO",                  # Ethanol
        "c1ccccc1",             # Benzene
        "CC(=O)O",              # Acetic acid
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    ]


@pytest.fixture
def sample_invalid_smiles():
    """Return invalid SMILES strings for testing error handling."""
    return [
        "INVALID",
        "XYZ123",
        "",
        "((()))",
    ]
