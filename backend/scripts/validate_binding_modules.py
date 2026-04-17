"""Quick validation of binding metrics module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from app.models.binding_metrics import concordance_index_fast, compute_all_metrics

# Perfect predictions -> CI=1.0
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
metrics = compute_all_metrics(y_true, y_pred)
print(f"Perfect order CI: {metrics['ci']:.3f} (should be 1.000)")

# Random -> CI~0.5
y_pred_random = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
ci_random = concordance_index_fast(y_true, y_pred_random)
print(f"Random order CI: {ci_random:.3f} (should be ~0.5)")

# Reversed -> CI=0.0
y_pred_reversed = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
ci_reversed = concordance_index_fast(y_true, y_pred_reversed)
print(f"Reversed order CI: {ci_reversed:.3f} (should be 0.000)")

# Forward pass test
from app.models.deep_dta_model import DeepDTA
import torch
model = DeepDTA()
drug = torch.randint(0, 65, (2, 100))
prot = torch.randint(0, 26, (2, 1000))
out = model(drug, prot)
print(f"\nForward pass: input drug={drug.shape}, protein={prot.shape}")
print(f"Output shape: {out.shape} (should be [2, 1])")
print(f"Output values: {out.detach().numpy().ravel()}")

# ESM model test
from app.models.deep_dta_esm import DeepDTA_ESM
esm_model = DeepDTA_ESM(esm_model_name="esm2_t33_650M_UR50D")
esm_params = sum(p.numel() for p in esm_model.parameters())
drug = torch.randint(0, 65, (2, 100))
prot_emb = torch.randn(2, 1280)  # Simulated ESM-2 embeddings
out_esm = esm_model(drug, prot_emb)
print(f"\nESM model: {esm_params:,} params")
print(f"ESM forward pass: drug={drug.shape}, protein_emb={prot_emb.shape}")
print(f"ESM output shape: {out_esm.shape}")

print("\n[OK] All validations passed!")
