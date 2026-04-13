"""
Tests for Molecule Service

Tests for SMILES tokenizer, VAE model, and molecule service functions.
"""

import pytest
from app.models.smiles_tokenizer import SmilesTokenizer
from app.services.molecule_service import validate_smiles, compute_molecule_properties


# ---------------------------------------------------------------------------
# Tokenizer Tests
# ---------------------------------------------------------------------------

class TestSmilesTokenizer:
    """Tests for the character-level SMILES tokenizer."""

    def setup_method(self):
        """Create a fresh tokenizer for each test."""
        self.tokenizer = SmilesTokenizer(max_length=120)

    def test_vocabulary_includes_special_tokens(self):
        """Vocabulary should include PAD, START, and END tokens."""
        assert self.tokenizer.pad_index == 0
        assert self.tokenizer.start_index == 1
        assert self.tokenizer.end_index == 2

    def test_vocabulary_size_is_positive(self):
        """Vocabulary size should be > 0."""
        assert self.tokenizer.vocabulary_size > 0

    def test_encode_produces_fixed_length(self):
        """Encoded sequences should always have length max_length + 2."""
        smiles = "CCO"
        encoded = self.tokenizer.encode(smiles)
        expected_length = self.tokenizer.max_length + 2
        assert len(encoded) == expected_length

    def test_encode_starts_with_start_token(self):
        """Encoded sequence should start with the START token."""
        encoded = self.tokenizer.encode("CCO")
        assert encoded[0] == self.tokenizer.start_index

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should return the original SMILES."""
        original_smiles = "CCO"
        encoded = self.tokenizer.encode(original_smiles)
        decoded = self.tokenizer.decode(encoded)
        assert decoded == original_smiles

    def test_encode_decode_complex_smiles(self):
        """Roundtrip should work for complex SMILES strings."""
        complex_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        encoded = self.tokenizer.encode(complex_smiles)
        decoded = self.tokenizer.decode(encoded)
        assert decoded == complex_smiles

    def test_batch_encode_decode(self):
        """Batch encoding and decoding should work correctly."""
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        encoded_batch = self.tokenizer.encode_batch(smiles_list)
        decoded_batch = self.tokenizer.decode_batch(encoded_batch)
        assert decoded_batch == smiles_list


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

class TestMoleculeValidation:
    """Tests for SMILES validation and property computation."""

    def test_valid_smiles_returns_true(self):
        """Valid SMILES should return is_valid=True."""
        result = validate_smiles("CCO")
        assert result["is_valid"] is True
        assert result["canonical_smiles"] is not None

    def test_invalid_smiles_returns_false(self):
        """Invalid SMILES should return is_valid=False."""
        result = validate_smiles("INVALID_STRING")
        assert result["is_valid"] is False
        assert result["canonical_smiles"] is None

    def test_valid_smiles_has_properties(self):
        """Valid SMILES should return computed properties."""
        result = validate_smiles("CCO")
        assert "molecular_weight" in result["properties"]
        assert "logp" in result["properties"]
        assert "qed_score" in result["properties"]

    def test_aspirin_properties(self):
        """Aspirin should have known properties."""
        result = validate_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert result["is_valid"] is True
        props = result["properties"]
        # Aspirin MW ≈ 180.16
        assert 179 < props["molecular_weight"] < 181


# ---------------------------------------------------------------------------
# API Route Tests
# ---------------------------------------------------------------------------

class TestMoleculeAPI:
    """Tests for molecule API endpoints."""

    def test_health_endpoint(self, api_client):
        """Health endpoint should return 200."""
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_validate_valid_smiles(self, api_client):
        """POST /molecules/validate with valid SMILES should succeed."""
        response = api_client.post(
            "/api/v1/molecules/validate",
            json={"smiles": "CCO"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["canonical_smiles"] == "CCO"

    def test_validate_invalid_smiles(self, api_client):
        """POST /molecules/validate with invalid SMILES should return is_valid=False."""
        response = api_client.post(
            "/api/v1/molecules/validate",
            json={"smiles": "NOT_A_MOLECULE"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
