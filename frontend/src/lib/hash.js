/**
 * SHA-256 of a value, as lowercase hex.
 *
 * The chain stores only this digest, never the molecule or the prediction,
 * so the same helper produces what gets recorded and what gets verified.
 */
export async function sha256(value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

/** The canonical record for one candidate, hashed for the audit trail. */
export function candidateRecord(molecule, target) {
  return {
    event: 'candidate',
    drug_smiles: molecule.smiles,
    protein: target,
    binding_affinity: molecule.affinity,
    toxicity_probability: molecule.toxicity,
    qed: molecule.properties?.qed_score ?? null,
  }
}
