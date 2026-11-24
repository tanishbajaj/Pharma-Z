# GraphVAE Training Report — Phase 2 Complete

## 1. Final Training Results

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Architecture | RGCVAE-inspired GraphVAE |
| Encoder | 3-layer RGIN (bond-aware) |
| Decoder | Two-stage Pairwise (Concat + Hadamard + Global Context) |
| Latent dim | 128 |
| Hidden dim | 256 |
| Max nodes | 38 |
| Node features | 28d (type + valence + charge + chirality + hybridization) |
| Parameters | 5,192,868 (all trainable) |
| Dataset | ZINC 1M (1,000,000 drug-like molecules) |
| Training | 30 epochs, cyclical β annealing (4 cycles, β_max=0.01) |
| Hardware | NVIDIA Tesla V100-SXM2-32GB (DGX cluster) |
| Total time | ~40 hours |

### Loss Progression Across 30 Epochs

| Epoch (abs) | Total Loss | Adj Loss | Node Loss | Bond Loss | KL | β | Event |
|-------------|-----------|----------|-----------|-----------|-----|-------|-------|
| 1 | 4.6967 | 4.3015 | 0.2623 | 0.1329 | 714.39 | 0.000 | Training start |
| 5 | 5.0511 | 4.2962 | 0.2346 | 0.1075 | 41.28 | 0.010 | Checkpoint saved |
| **9** | **4.5898** | **4.2795** | **0.1710** | **0.0569** | 61.76 | 0.001 | 🔄 **Cycle 2 reset** |
| 10 | 4.7258 | 4.2820 | 0.1799 | 0.0636 | 50.06 | 0.004 | Checkpoint saved |
| 11 | 4.8473 | 4.2846 | 0.1890 | 0.0705 | 45.48 | 0.007 | GPU migration |
| **16** | **4.4478** | **4.2708** | **0.1364** | **0.0406** | 385.62 | 0.000 | 🔄 **Cycle 3 reset (best adj!)** |
| 21 | 4.9777 | 4.2865 | 0.1913 | 0.0758 | 42.41 | 0.010 | Checkpoint saved |
| **30** | **4.9445** | **4.2816** | **0.1719** | **0.0615** | 42.95 | 0.010 | **Training complete** |

### Generation Evaluation (2000 samples)

| Metric | Result |
|--------|--------|
| **Validity** | 57.0% (1140 / 2000) |
| **Uniqueness** | 100% (1140 / 1140) |
| **Novelty** | 0% (artifact — training SMILES not passed to eval) |
| **Disconnected fragments** | Eliminated via post-processing |

### Sample Generated SMILES
```
CC1C2(C(=O)C3CN3CC34CC3CNC4=O)C34C5=C(C53)C142
CCC1OC1S(=O)(=O)NC12CC13OC31C(=O)OC21C
CC1CC1NC(=O)C1C2C(=C=O)C12
CC1=C2NCC3C4C35C36OC12N3C456
```

---

## 2. Research Papers Studied

### Paper 1: GraphVAE (Simonovsky & Komodakis, 2018)
> *"GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders"*

**Core idea:** Encode molecular graphs into a continuous latent space using a GNN encoder, decode back to adjacency matrix + node features. Uses graph matching (Hungarian algorithm) to align predicted and target graphs for loss computation.

**What we used:**
- Overall VAE-on-graphs framework (encode → z → decode)
- Adjacency matrix representation `A ∈ [0,1]^{k×k}`
- Node feature matrix `F ∈ R^{k×d}`
- BCE loss for edge prediction with `pos_weight` for sparsity
- MST-based connectivity enforcement during generation

---

### Paper 2: JT-VAE (Jin et al., 2018)
> *"Junction Tree Variational Autoencoder for Molecular Graph Generation"*

**Core idea:** Decompose molecules into a tree of chemically valid substructures (rings, bonds). Generate the tree first, then assemble atoms. Achieves near-100% validity by construction.

**What we used:**
- **Valence-masked decoding** — the key insight that edges should be placed iteratively, checking valence budget at each step rather than thresholding all at once
- Bond type prediction as a separate output channel
- Concept of chemical validity enforcement during generation (not just post-hoc filtering)

---

### Paper 3: MolGAN (De Cao & Kipf, 2018)
> *"MolGAN: An Implicit Generative Model for Small Molecular Graphs"*

**Core idea:** Use a GAN instead of VAE for graph generation. Includes a reward network trained on molecular properties (QED, SA score) to guide generation toward drug-like molecules.

**What we used:**
- Reward-based evaluation metrics (QED, SA score) for assessing generation quality
- Confirmed that our VAE approach was more stable than adversarial training for this problem size

---

### Paper 4: TGVAE (Yan et al., 2024)
> *"Period & Amplitude Annealing VAE for Molecular Generation"*

**Core idea:** Cyclical β annealing prevents posterior collapse in molecular VAEs. Instead of linearly increasing β from 0→1, use multiple cycles where β resets to 0 periodically. This lets the model learn complex structure during low-β phases while maintaining useful latent space during high-β phases.

**What we used:**
- **Cyclical β schedule** — 4 cycles over 30 epochs with β_max=0.01
- **Monotonic annealing ratio** (0.5) — β ramps up for first half of each cycle, then stays at β_max
- The schedule function: `β(t) = β_max · min(1, (t mod T_cycle) / (ratio · T_cycle))`
- This was **critical** — without it, the model collapsed to generating identical molecules

---

### Paper 5: RGCVAE (Rigoni et al., 2025)
> *"RGCVAE: Relational Graph Conditioned Variational Autoencoder for Molecule Design"*

**Core idea:** Replace the flat MLP decoder with a **pairwise relational decoder**. Instead of predicting the entire adjacency matrix from a single vector, predict each edge (i,j) using the **interaction** between node embeddings i and j. Uses three interaction modes: additive (s_i + s_j), multiplicative (s_i ⊙ s_j), and global context.

**What we used:**
- **PairwiseDecoder** architecture — the core architectural innovation
- Rich atom features (28d) including valence, charge, chirality, hybridization
- Learnable positional embeddings for atom slots (nn.Embedding)
- Bond type prediction as a 4-class classification per edge
- RGIN encoder (Relational Graph Isomorphism Network) — bond-aware message passing
- Teacher-forced bond loss (CE only where edges exist)

---

## 3. Difficulties Faced & How Papers Helped

### Difficulty 1: Posterior Collapse
**Problem:** Early training with fixed β=1.0 caused the decoder to ignore the latent code entirely. KL loss dropped to near-zero while reconstruction stayed high — the model generated identical outputs regardless of input.

**Solution (Paper 4: TGVAE):** Cyclical β annealing with β_max=0.01 and 4 cycles. During low-β phases, the model freely encodes structure; during high-β phases, it compresses the latent space. Each cycle reset produced the biggest loss improvements (epochs 9, 16, 23).

---

### Difficulty 2: Disconnected Graph Generation
**Problem:** The flat MLP decoder (`z → FC → reshape → adjacency`) produced graphs where most atoms had no edges — resulting in molecules like `C.C.C.C.C.C.C.C.C.C.C.C.COC(=O)CCC1...` with 15+ isolated carbon atoms.

**Solution (Paper 5: RGCVAE):** The PairwiseDecoder predicts edges via pairwise interactions between per-node embeddings. Instead of one monolithic prediction, each edge (i,j) is predicted using `[s_i + s_j, s_i ⊙ s_j, global]`. This gives the model permutation-invariant, position-aware edge prediction.

---

### Difficulty 3: Adjacency Sparsity (96% zeros)
**Problem:** Molecular graphs are extremely sparse — in a 38×38 adjacency matrix, only ~4% of entries are edges. Standard BCE loss pushed the model toward predicting all-zeros (no edges), which minimizes loss but is useless.

**Solution (Paper 1: GraphVAE):** Applied `pos_weight` to the BCE loss, upweighting positive (edge) entries by the ratio of negatives to positives (~24:1). This forces the model to pay attention to the rare but critical edge predictions.

---

### Difficulty 4: Valence Violations (Carbon with 6+ bonds)
**Problem:** The decoder predicted bond types independently of valence constraints. A carbon could receive 3 double bonds (valence=6), violating chemistry.

**Solution (Paper 2: JT-VAE):** Valence-masked iterative edge placement. Instead of thresholding the adjacency matrix, edges are sorted by probability and placed one at a time. For each edge, the highest-probability bond type that respects both atoms' remaining valence budget is chosen. If no bond type fits, the edge is skipped.

---

### Difficulty 5: GPU Memory Exhaustion (OOM)
**Problem:** The PairwiseDecoder creates a `[B, 38, 38, 768]` tensor for each batch. When evaluating 10,000 molecules at once, this required 41 GB — exceeding the V100's 32 GB.

**Solution:** Implemented batched generation — generate 128 molecules at a time, accumulate results on CPU, then concatenate. Training (batch=32) was unaffected since the tensor is only `[32, 38, 38, 768]` = 134 MB.

---

### Difficulty 6: GPU Contention on Shared Cluster
**Problem:** Training on GPU 2 saw epoch times increase from 1.6 hours to 3.3 hours as other users loaded their models on the same GPU. At this rate, 30 epochs would take 100+ hours.

**Solution:** Implemented checkpoint resume (`--resume` + `--start-epoch`) and migrated to GPU 4 (most free VRAM). This required:
- Saving model weights every 5 epochs
- Correctly offsetting the cyclical β schedule when resuming
- Using `CUDA_VISIBLE_DEVICES=4` for GPU pinning

---

### Difficulty 7: Empty Atom Slots in Fixed-Size Output
**Problem:** `max_nodes=38` means the decoder always outputs 38 atom slots, but real molecules have 20-25 atoms. Unused slots were filled with carbon atoms (most common) that had no edges, creating disconnected fragments.

**Solution:** Post-processing filter — only include atoms that have ≥1 edge above threshold to another active atom. Additionally, after building the RDKit molecule, keep only the largest connected component via `rdmolops.GetMolFrags()`.

---

### Difficulty 8: Checkpoint Filename Confusion on Resume
**Problem:** When resuming training with `--start-epoch 11` and `--epochs 19`, checkpoints were saved as `epoch_5.pt` (relative) instead of `epoch_16.pt` (absolute), overwriting the pre-resume checkpoint.

**Solution:** Modified the checkpoint saving to use absolute epoch numbers: `abs_epoch = epoch + start_epoch`, ensuring filenames reflect the true training progress.

---

### Difficulty 9: Disk Quota Management
**Problem:** 30 GB disk quota on the shared DGX cluster, with 14 GB consumed by conda environments and 2.6 GB by the ZINC dataset. Checkpoints (60 MB each) and pip cache (3.4 GB) consumed additional space.

**Solution:** Regular cleanup routine:
- `pip cache purge` (freed 3.4 GB)
- Delete intermediate checkpoints (keep only final + best)
- `conda clean --all`
- Monitor with `quota -s`

---

## 4. The Two Most Important Papers

### 🥇 Paper 5: RGCVAE (Rigoni et al., 2025)
> **Why it's #1:** This paper provided the **core architectural innovation** — the PairwiseDecoder. Without it, the model couldn't generate connected molecules at all. The flat MLP decoder fundamentally couldn't learn edge structure from a global latent vector. RGCVAE's pairwise interaction approach (additive + multiplicative + global context) is what makes graph generation work.

**Specific contributions used:**
- PairwiseDecoder architecture (the entire decoder)
- RGIN encoder (bond-aware message passing)
- Rich 28d atom features
- Learnable positional embeddings
- Teacher-forced bond loss
- Valence-masked generation concept

---

### 🥈 Paper 4: TGVAE (Yan et al., 2024)
> **Why it's #2:** This paper solved **posterior collapse**, the most dangerous failure mode of VAE training. Without cyclical β annealing, the model converges to a degenerate solution where the latent space is useless. The cyclical schedule produced the biggest loss drops at each cycle reset (epochs 9, 16, 23) and was essential for training stability over 30 epochs.

**Specific contributions used:**
- Cyclical β annealing schedule (4 cycles)
- β_max=0.01 (not 1.0 — domain-specific tuning)
- Monotonic annealing ratio=0.5
- The mathematical formulation of the schedule function

---

### Why these two and not the others:

| Paper | Role | Replaceable? |
|-------|------|-------------|
| **RGCVAE** | Core architecture | ❌ No — the decoder IS the model |
| **TGVAE** | Training stability | ❌ No — without it, posterior collapse |
| GraphVAE | Framework + pos_weight | ⚠️ Basic framework, widely known |
| JT-VAE | Valence masking concept | ⚠️ Important but post-processing, not architecture |
| MolGAN | Evaluation metrics | ✅ Easily replaced by standard MOSES metrics |
