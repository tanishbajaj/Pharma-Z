"""
Pipeline Service

Orchestrates the full drug discovery pipeline:
    1. Generate molecules (or accept input SMILES)
    2. Predict binding affinity
    3. Predict toxicity
    4. Rank candidates
"""

from typing import List, Optional

from app.services import molecule_service
from app.services import binding_service
from app.services import toxicity_service
from app.services import ranking_service
from app.utils.logging_config import logger


def run_full_pipeline(
    smiles_input: str,
    protein_sequence: str,
    num_generated: int = 0,
) -> dict:
    """
    Run the complete drug discovery pipeline for a single drug-protein pair.

    Args:
        smiles_input:     SMILES string of the drug candidate.
        protein_sequence: Target protein amino acid sequence.
        num_generated:    If > 0, also generate additional molecules and rank all.

    Returns:
        Dict with binding, toxicity, score, and optionally generated candidates.
    """
    logger.info(f"Running pipeline: drug={smiles_input[:30]}..., protein={protein_sequence[:30]}...")

    # Step 1: Validate input molecule
    validation = molecule_service.validate_smiles(smiles_input)
    if not validation["is_valid"]:
        return {
            "status": "failed",
            "message": f"Invalid SMILES: {smiles_input}",
            "smiles": smiles_input,
        }

    drug_likeness = validation["properties"].get("qed_score", 0.5)

    # Step 2: Predict binding affinity
    binding_result = binding_service.predict_binding_affinity(
        drug_smiles=smiles_input,
        protein_sequence=protein_sequence,
    )
    binding_score = binding_result["binding_affinity_score"]

    # Step 3: Predict toxicity
    toxicity_result = toxicity_service.predict_toxicity(smiles_input)
    toxicity_score = toxicity_result["toxicity_probability"]

    # Step 4: Compute ranking score
    final_score = ranking_service.compute_candidate_score(
        binding_affinity=binding_score,
        toxicity_probability=toxicity_score,
        drug_likeness=drug_likeness,
    )

    result = {
        "status": "completed",
        "smiles": smiles_input,
        "canonical_smiles": validation["canonical_smiles"],
        "protein_sequence": protein_sequence[:50] + "..." if len(protein_sequence) > 50 else protein_sequence,
        "binding_affinity": binding_score,
        "toxicity_probability": toxicity_score,
        "is_toxic": toxicity_result["is_toxic"],
        "drug_likeness": drug_likeness,
        "final_score": final_score,
        "properties": validation["properties"],
        "ranking_weights": ranking_service.get_ranking_weights(),
    }

    # Optional: generate additional molecules and rank them
    if num_generated > 0:
        generated = molecule_service.generate_molecules(num_molecules=num_generated)
        candidates = []

        for mol in generated:
            if mol["is_valid"]:
                try:
                    mol_binding = binding_service.predict_binding_affinity(
                        drug_smiles=mol["smiles"],
                        protein_sequence=protein_sequence,
                    )
                    mol_toxicity = toxicity_service.predict_toxicity(mol["smiles"])

                    candidates.append({
                        "smiles": mol["smiles"],
                        "binding_affinity": mol_binding["binding_affinity_score"],
                        "toxicity_probability": mol_toxicity["toxicity_probability"],
                        "drug_likeness": mol["properties"].get("qed_score", 0.5),
                    })
                except Exception as error:
                    logger.warning(f"Pipeline failed for {mol['smiles']}: {error}")

        # Add input molecule to candidates
        candidates.append({
            "smiles": smiles_input,
            "binding_affinity": binding_score,
            "toxicity_probability": toxicity_score,
            "drug_likeness": drug_likeness,
        })

        ranked_candidates = ranking_service.rank_candidates(candidates)
        result["ranked_candidates"] = ranked_candidates
        result["total_candidates"] = len(ranked_candidates)

    logger.info(
        f"Pipeline result: binding={binding_score:.4f}, "
        f"toxicity={toxicity_score:.4f}, score={final_score:.4f}"
    )

    return result


def run_batch_pipeline(
    drug_smiles_list: List[str],
    protein_sequence: str,
) -> List[dict]:
    """
    Run the pipeline for multiple drug candidates against a single protein target.

    Args:
        drug_smiles_list: List of drug SMILES strings.
        protein_sequence: Target protein sequence.

    Returns:
        Ranked list of candidate results.
    """
    logger.info(f"Running batch pipeline: {len(drug_smiles_list)} candidates")

    candidates = []
    for smiles in drug_smiles_list:
        result = run_full_pipeline(smiles, protein_sequence)
        if result["status"] == "completed":
            candidates.append({
                "smiles": smiles,
                "binding_affinity": result["binding_affinity"],
                "toxicity_probability": result["toxicity_probability"],
                "drug_likeness": result["drug_likeness"],
            })

    ranked = ranking_service.rank_candidates(candidates)
    logger.info(f"Batch pipeline complete: {len(ranked)} valid candidates ranked")
    return ranked
