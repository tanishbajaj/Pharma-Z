"""
Ranking Service

Computes final drug candidate scores using a weighted formula:
    score = w1 * binding_affinity - w2 * toxicity + w3 * drug_likeness

Higher scores indicate better drug candidates.
"""

from typing import List, Optional

from app.utils.constants import (
    RANKING_WEIGHT_BINDING,
    RANKING_WEIGHT_TOXICITY,
    RANKING_WEIGHT_DRUGLIKENESS,
)
from app.utils.logging_config import logger


# ---------------------------------------------------------------------------
# Global Weights (can be updated via API)
# ---------------------------------------------------------------------------

_ranking_weights = {
    "binding": RANKING_WEIGHT_BINDING,
    "toxicity": RANKING_WEIGHT_TOXICITY,
    "druglikeness": RANKING_WEIGHT_DRUGLIKENESS,
}


def get_ranking_weights() -> dict:
    """Return the current ranking weights."""
    return dict(_ranking_weights)


def update_ranking_weights(
    weight_binding: float = None,
    weight_toxicity: float = None,
    weight_druglikeness: float = None,
) -> dict:
    """Update ranking weights and return the new values."""
    if weight_binding is not None:
        _ranking_weights["binding"] = weight_binding
    if weight_toxicity is not None:
        _ranking_weights["toxicity"] = weight_toxicity
    if weight_druglikeness is not None:
        _ranking_weights["druglikeness"] = weight_druglikeness

    logger.info(f"Ranking weights updated: {_ranking_weights}")
    return dict(_ranking_weights)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_candidate_score(
    binding_affinity: float,
    toxicity_probability: float,
    drug_likeness: float = 0.5,
    weights: dict = None,
) -> float:
    """
    Compute a final ranking score for a drug candidate.

    Formula: score = w1 * binding - w2 * toxicity + w3 * druglikeness

    Args:
        binding_affinity:    Predicted binding affinity score (higher = better).
        toxicity_probability: Predicted toxicity probability (0–1, lower = safer).
        drug_likeness:       QED score or similar (0–1, higher = more drug-like).
        weights:             Custom weights dict (uses defaults if None).

    Returns:
        Final candidate score (float).
    """
    if weights is None:
        weights = _ranking_weights

    w1 = weights.get("binding", RANKING_WEIGHT_BINDING)
    w2 = weights.get("toxicity", RANKING_WEIGHT_TOXICITY)
    w3 = weights.get("druglikeness", RANKING_WEIGHT_DRUGLIKENESS)

    score = (w1 * binding_affinity) - (w2 * toxicity_probability) + (w3 * drug_likeness)
    return round(score, 4)


def rank_candidates(candidates: List[dict]) -> List[dict]:
    """
    Rank a list of drug candidates by their computed scores.

    Each candidate dict must have: binding_affinity, toxicity_probability,
    and optionally drug_likeness.

    Args:
        candidates: List of candidate dicts.

    Returns:
        Sorted list with rank and score added to each candidate.
    """
    scored_candidates = []
    for candidate in candidates:
        score = compute_candidate_score(
            binding_affinity=candidate.get("binding_affinity", 0.0),
            toxicity_probability=candidate.get("toxicity_probability", 0.0),
            drug_likeness=candidate.get("drug_likeness", 0.5),
        )
        candidate_with_score = {**candidate, "final_score": score}
        scored_candidates.append(candidate_with_score)

    # Sort descending by score (higher = better candidate)
    scored_candidates.sort(key=lambda c: c["final_score"], reverse=True)

    # Add rank
    for rank, candidate in enumerate(scored_candidates, start=1):
        candidate["rank"] = rank

    logger.info(f"Ranked {len(scored_candidates)} drug candidates")
    return scored_candidates
