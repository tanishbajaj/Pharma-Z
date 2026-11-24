"""
Select top 100 molecules from generated output most likely to bind GSK3B (Diabetes).

Kinase inhibitor profile (from KIBA training data analysis):
  - MW:     250–450 Da
  - LogP:   1.0–4.5  (moderately lipophilic)
  - TPSA:   30–90 Å²  (membrane permeable)
  - H-donors: ≤ 3
  - H-acceptors: ≤ 7
  - QED:    > 0.75 (highly drug-like)
  - Rings:  ≥ 1
"""

import json
import sys

# ── GSK3B protein sequence (UniProt P49841) ──────────────────────────────────
GSK3B_SEQUENCE = (
    "MSGRPRTTSFAESCKPVQQPSAFGSMKVSRDKDGSKVTTVVATPGQGPDRPQEVSYTDTKVIGNGSFGVVY"
    "QAKLCDSGELVAIKKV LQDKRFKNRELQIMRKL DHCNIVRLRYFFYSSGEKKDEVYLNLVLDYVPETVYRVA"
    "RHYSRAKQTLPVIYVKLYMYQLFRSLAYI HSFGICHRDIKPQNLLLDPDTAVLKLCDFGSAKQLVRGEPNVSY"
    "ICSRYYRAPELI FGATDYTSSIDVWSAGCVLAELLL GQPIFPGDSGVDQLVEIIKVLGTPTREQIREMNPNYTE"
    "FKFPQIKAHPWTKVFRPRTPPEAIALCSRLLEYTPTARLTPLEACAHSFFD ELRDPNVKLPNGRDTPALFNFTT"
    "QELSS NPPLATILIPPHARIQAAASTPTN ATAASDAN TGDRGQTNN AASASASNS T"
).replace(" ", "")

# ── Load molecules ────────────────────────────────────────────────────────────
input_file = "generated_molecules.json"
try:
    with open(input_file) as f:
        data = json.load(f)
    molecules = data.get("molecules", data)
except FileNotFoundError:
    print(f"ERROR: Save the API response as '{input_file}' in the backend folder first.")
    sys.exit(1)

# ── Filter valid molecules ────────────────────────────────────────────────────
valid = [m for m in molecules if m.get("is_valid") and m.get("properties")]
print(f"Total generated : {len(molecules)}")
print(f"Valid molecules : {len(valid)}")

# ── Score each molecule by kinase-inhibitor profile ───────────────────────────
def kinase_score(mol):
    p = mol["properties"]
    mw    = p.get("molecular_weight", 0)
    logp  = p.get("logp", -99)
    tpsa  = p.get("tpsa", 999)
    hd    = p.get("num_h_donors", 99)
    ha    = p.get("num_h_acceptors", 99)
    qed   = p.get("qed_score", 0)
    rings = p.get("num_rings", 0)

    score = 0.0

    # MW: 250-450 ideal
    if 250 <= mw <= 450:
        score += 20
    elif 200 <= mw < 250 or 450 < mw <= 500:
        score += 10

    # LogP: 1.0-4.5 ideal for kinase inhibitors
    if 1.0 <= logp <= 4.5:
        score += 25
    elif 0.5 <= logp < 1.0 or 4.5 < logp <= 5.0:
        score += 10

    # TPSA: 30-90 ideal
    if 30 <= tpsa <= 90:
        score += 20
    elif 20 <= tpsa < 30 or 90 < tpsa <= 120:
        score += 10

    # H-donors: ≤ 3
    if hd <= 3:
        score += 10
    elif hd <= 5:
        score += 5

    # H-acceptors: ≤ 7
    if ha <= 7:
        score += 10
    elif ha <= 10:
        score += 5

    # QED (0-1): highest weight
    score += qed * 35

    # Must have rings (kinases bind aromatic/cyclic compounds)
    if rings >= 2:
        score += 10
    elif rings == 1:
        score += 5

    return score

# Rank by kinase score
ranked = sorted(valid, key=kinase_score, reverse=True)
top100 = ranked[:100]

print(f"\nSelected top {len(top100)} candidates by kinase-inhibitor profile\n")
print("Top 10 preview:")
for i, mol in enumerate(top100[:10], 1):
    p = mol["properties"]
    ks = kinase_score(mol)
    print(
        f"  {i:2}. QED={p.get('qed_score', 0):.3f} | "
        f"MW={p.get('molecular_weight', 0):.1f} | "
        f"LogP={p.get('logp', 0):.2f} | "
        f"TPSA={p.get('tpsa', 0):.1f} | "
        f"KScore={ks:.1f} | "
        f"{mol['smiles'][:55]}"
    )

# ── Build batch-predict JSON ──────────────────────────────────────────────────
smiles_list   = [m["smiles"] for m in top100]
protein_list  = [GSK3B_SEQUENCE] * len(smiles_list)

batch_payload = {
    "drug_smiles_list": smiles_list,
    "protein_sequence_list": protein_list
}

output_file = "top100_gsk3b_batch.json"
with open(output_file, "w") as f:
    json.dump(batch_payload, f, indent=2)

print(f"\n✅ Saved batch-predict JSON → {output_file}")
print(f"   {len(smiles_list)} drug SMILES × 1 protein (GSK3B / Type 2 Diabetes)")
print(f"\nNow POST this file to:")
print(f"  http://localhost:8000/api/v1/binding/batch-predict")
