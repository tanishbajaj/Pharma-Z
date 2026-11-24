"""
Phase 5: Toxicity Model Training (Random Forest on Morgan fingerprints)

Trains the Tox21 multi-label toxicity classifier used by the ranking
pipeline. CPU-only — no GPU required.

Usage:
    python scripts/train_toxicity.py --split scaffold
    python scripts/train_toxicity.py --split random      # optimistic baseline
    python scripts/train_toxicity.py --split both        # report both

Method, and why:

  Masked training (Ebner et al. 2025, arXiv:2511.14744)
      Tox21 measures each molecule on only a subset of the 12 assays, so
      ~17% of label cells are missing. Filling them with 0 invents ~16,000
      "non-toxic" labels. That paper identifies zero-filling as a source of
      benchmark drift on Tox21. We train one RF per assay on the measured
      rows only.

  Scaffold split (Tursunbadalov & Tursunbadalov, arXiv:2607.06605)
      Bemis-Murcko scaffold groups are assigned whole to one partition, so
      structural analogs cannot leak between train and test. Expect a lower
      AUC than a random split — the ToxBench project page reports a mean
      Tox21 random-to-scaffold macro-AUROC drop of 0.070, and 0.057 for a
      Random Forest specifically.

  Validation-chosen aggregation and threshold (ToxLens, arXiv:2608.30472)
      That work selects decision thresholds with a "validation-based
      threshold-selection protocol". We pick both the aggregation rule
      (mean / max / noisy-OR) and the cutoff on validation, never on test.

  No post-hoc calibration
      The ToxBench page reports Tox21 random forests are "already well
      calibrated at ECE 0.018 and gain nothing from correction", so Platt
      and isotonic scaling are deliberately not applied here.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

from app.config import settings
from app.models.toxicity_model import ToxicityPredictor, batch_smiles_to_fingerprints
from app.services.toxicity_service import TOX21_LABEL_COLUMNS, scaffold_split
from app.utils.constants import TOX21_FILENAME

AGGREGATIONS = ("max", "noisy_or", "mean")


def aggregate(probabilities: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return probabilities.max(axis=1)
    if mode == "noisy_or":
        return 1.0 - np.prod(1.0 - probabilities, axis=1)
    return probabilities.mean(axis=1)


def per_assay_auc(model, features, labels, assays):
    """ROC-AUC / PR-AUC per assay on measured labels only."""
    probabilities = model.predict_proba(features)
    rows, aucs = {}, []
    for index, assay in enumerate(assays):
        measured = ~np.isnan(labels[:, index])
        y_true = labels[measured, index]
        y_score = probabilities[measured, index]
        if len(np.unique(y_true)) < 2:
            continue
        auc = roc_auc_score(y_true, y_score)
        rows[assay] = {
            "n_test": int(measured.sum()),
            "positive_rate": round(float(y_true.mean()), 4),
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        }
        aucs.append(auc)
    return rows, float(np.mean(aucs)), probabilities


def choose_gate(model, features, labels, assays):
    """
    Pick aggregation rule + threshold on VALIDATION data.

    Target: catch molecules positive on any assay.

    Selection uses Youden's J (TPR - FPR), not F1. F1 is maximised by a gate
    that flags everything whenever the positive rate is high — which is the
    mirror image of the old mean>0.5 gate that flagged nothing. J is 0 for
    both degenerate gates, so it can only be won by a cutoff that actually
    separates the two classes.
    """
    probabilities = model.predict_proba(features)
    any_positive = (np.nan_to_num(labels, nan=0.0).sum(axis=1) >= 1).astype(int)

    best = None
    table = []
    for mode in AGGREGATIONS:
        score = aggregate(probabilities, mode)
        auc = roc_auc_score(any_positive, score)
        for threshold in np.arange(0.05, 0.96, 0.01):
            flagged = score > threshold
            tpr = flagged[any_positive == 1].mean() if (any_positive == 1).any() else 0.0
            fpr = flagged[any_positive == 0].mean() if (any_positive == 0).any() else 0.0
            youden = tpr - fpr
            if best is None or youden > best["youden_j"]:
                precision = any_positive[flagged].mean() if flagged.sum() else 0.0
                best = {
                    "aggregation": mode,
                    "threshold": round(float(threshold), 2),
                    "precision": round(float(precision), 3),
                    "recall": round(float(tpr), 3),
                    "false_positive_rate": round(float(fpr), 3),
                    "youden_j": round(float(youden), 3),
                    "flagged_fraction": round(float(flagged.mean()), 3),
                }
        table.append((mode, round(float(auc), 4)))
    return best, table


def run(frame, assays, split_mode, args, seed=None, features=None, labels=None, smiles=None):
    seed = args.seed if seed is None else seed
    print(f"\n{'=' * 68}")
    print(f"SPLIT: {split_mode.upper()}   seed={seed}")
    print(f"{'=' * 68}")

    if features is None:
        smiles = frame["smiles"].tolist()
        labels_all = frame[assays].values.astype(np.float32)  # NaN preserved
        features, valid_idx, invalid_idx = batch_smiles_to_fingerprints(smiles)
        labels = labels_all[valid_idx]
        smiles = [smiles[i] for i in valid_idx]
        print(f"  Fingerprints: {len(valid_idx)} valid, {len(invalid_idx)} invalid")

    if split_mode == "scaffold":
        # Deterministic: scaffold groups are assigned largest-first, so across
        # seeds only the RF random_state varies. The resulting spread measures
        # model variance, NOT split variance.
        train_idx, valid_split, test_idx = scaffold_split(smiles, 0.8, 0.1)
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(features))
        n_test = int(len(order) * 0.1)
        test_idx = order[:n_test]
        valid_split = order[n_test: 2 * n_test]
        train_idx = order[2 * n_test:]

    print(f"  Train {len(train_idx)}  Valid {len(valid_split)}  Test {len(test_idx)}")

    model = ToxicityPredictor(n_estimators=args.n_estimators, random_state=seed)
    start = time.time()
    model.train_masked(features[train_idx], labels[train_idx], label_names=assays)
    elapsed = time.time() - start
    print(f"  Trained {len(assays)} masked RFs in {elapsed:.1f}s")
    print(f"  Rows per assay: min {min(model.model.n_train_per_assay)}, "
          f"max {max(model.model.n_train_per_assay)}")

    # --- Gate selection on VALIDATION ---
    gate, agg_table = choose_gate(
        model, features[valid_split], labels[valid_split], assays
    )
    print(f"\n  Aggregation ranking (validation AUC, 'any assay positive'):")
    for mode, auc in sorted(agg_table, key=lambda r: -r[1]):
        print(f"    {mode:<10} {auc:.4f}")
    print(f"  Chosen gate: {gate['aggregation']} > {gate['threshold']}  "
          f"(recall {gate['recall']}, FPR {gate['false_positive_rate']}, "
          f"J {gate['youden_j']}, flags {100 * gate['flagged_fraction']:.0f}%)")

    model.aggregation = gate["aggregation"]
    model.toxic_threshold = gate["threshold"]

    # --- Test evaluation ---
    rows, mean_auc, _ = per_assay_auc(
        model, features[test_idx], labels[test_idx], assays
    )
    print(f"\n  {'assay':<16}{'n':>7}{'pos%':>8}{'ROC-AUC':>10}{'PR-AUC':>9}")
    print(f"  {'-' * 50}")
    for assay, row in rows.items():
        print(f"  {assay:<16}{row['n_test']:>7}{100 * row['positive_rate']:>7.1f}%"
              f"{row['roc_auc']:>10.4f}{row['pr_auc']:>9.4f}")
    print(f"  {'-' * 50}")
    print(f"  {'MEAN':<16}{'':<15}{mean_auc:>10.4f}")

    # Does the gate actually fire on the test set?
    probabilities = model.predict_proba(features[test_idx])
    score = aggregate(probabilities, model.aggregation)
    flagged = (score > model.toxic_threshold).sum()
    print(f"\n  Gate on test: flags {flagged}/{len(test_idx)} "
          f"({100 * flagged / len(test_idx):.1f}%) as toxic")

    return model, {
        "split": split_mode,
        "train_size": int(len(train_idx)),
        "valid_size": int(len(valid_split)),
        "test_size": int(len(test_idx)),
        "mean_roc_auc": round(mean_auc, 4),
        "per_assay": rows,
        "gate": gate,
        "aggregation_validation_auc": dict(agg_table),
        "test_flagged_fraction": round(float(flagged / len(test_idx)), 4),
        "training_time_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the Tox21 toxicity model")
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(settings.data_raw_dir, TOX21_FILENAME))
    parser.add_argument("--split", choices=["scaffold", "random", "both"],
                        default="scaffold",
                        help="scaffold = honest; random = optimistic baseline")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Repeat with consecutive seeds and report mean +/- std")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"\n{'=' * 68}")
    print("Phase 5: Toxicity Model Training (masked RF / Tox21)")
    print(f"{'=' * 68}")

    if not os.path.exists(args.dataset):
        print(f"  x Not found: {args.dataset}")
        print(f"    curl -sL https://deepchemdata.s3-us-west-1.amazonaws.com"
              f"/datasets/tox21.csv.gz | gzip -dc > {args.dataset}")
        sys.exit(1)

    frame = pd.read_csv(args.dataset).dropna(subset=["smiles"])
    if args.max_samples:
        frame = frame.head(args.max_samples)
    assays = [c for c in TOX21_LABEL_COLUMNS if c in frame.columns]

    measured = frame[assays].notna().values
    print(f"\n  Molecules: {len(frame)}   Assays: {len(assays)}")
    print(f"  Measured label cells: {measured.sum()}/{measured.size} "
          f"({100 * measured.mean():.1f}%) — the rest are MASKED, not zero-filled")

    # Fingerprint once, reuse across every seed and split.
    all_smiles = frame["smiles"].tolist()
    all_labels = frame[assays].values.astype(np.float32)
    features, valid_idx, invalid_idx = batch_smiles_to_fingerprints(all_smiles)
    labels = all_labels[valid_idx]
    smiles = [all_smiles[i] for i in valid_idx]
    print(f"  Fingerprints: {len(valid_idx)} valid, {len(invalid_idx)} invalid")

    modes = ["random", "scaffold"] if args.split == "both" else [args.split]
    seeds = [args.seed + i for i in range(args.n_seeds)]
    results, keep_model = {}, None

    for mode in modes:
        per_seed = []
        for seed in seeds:
            model, result = run(frame, assays, mode, args, seed, features, labels, smiles)
            per_seed.append(result)
            if keep_model is None or (mode == "scaffold" and seed == seeds[0]):
                keep_model = model

        aucs = [r["mean_roc_auc"] for r in per_seed]
        summary = dict(per_seed[0])
        summary["n_seeds"] = len(seeds)
        summary["mean_roc_auc"] = round(float(np.mean(aucs)), 4)
        summary["std_roc_auc"] = round(float(np.std(aucs)), 4)
        summary["per_seed_roc_auc"] = [round(a, 4) for a in aucs]
        summary["seed_variance_note"] = (
            "scaffold split is deterministic; spread reflects RF seed only"
            if mode == "scaffold" else
            "random split varies with seed; spread reflects split + model"
        )
        results[mode] = summary

    save_dir = os.path.join(settings.saved_models_dir, "toxicity")
    os.makedirs(save_dir, exist_ok=True)
    save_path = args.output or os.path.join(save_dir, "toxicity_rf_model.pkl")
    joblib.dump(keep_model, save_path)

    payload = {
        "dataset": "tox21",
        "n_molecules": int(len(frame)),
        "n_assays": len(assays),
        "n_estimators": args.n_estimators,
        "fingerprint": "Morgan radius=2, 2048 bits",
        "missing_label_handling": "masked (per-assay RF on measured rows only)",
        "calibration": "none — ToxBench reports Tox21 RF already calibrated (ECE 0.018)",
        "shipped_model_split": "scaffold" if "scaffold" in results else modes[0],
        "results": results,
        "references": {
            "masking": "Ebner et al. 2025, arXiv:2511.14744 (Tox21 leaderboard); "
                       "RF overall AUC 0.851 on the ORIGINAL Challenge split — "
                       "different split, not directly comparable to these numbers",
            "scaffold_split": "Tursunbadalov & Tursunbadalov, arXiv:2607.06605",
            "threshold_protocol": "ToxLens, arXiv:2608.30472 (validation-based)",
            "split_effect_size": "ToxBench project page: RF Tox21 random->scaffold "
                                 "macro-AUROC drop 0.057 (not peer reviewed)",
        },
    }
    with open(os.path.join(save_dir, "training_results.json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\n{'=' * 68}")
    for mode, result in results.items():
        print(f"  {mode:<10} mean ROC-AUC {result['mean_roc_auc']:.4f} "
              f"+/- {result['std_roc_auc']:.4f}  (n={result['n_seeds']} seeds)   "
              f"gate {result['gate']['aggregation']} > {result['gate']['threshold']}")
    if len(results) == 2:
        drop = results["random"]["mean_roc_auc"] - results["scaffold"]["mean_roc_auc"]
        print(f"  random -> scaffold drop: {drop:.4f} "
              f"(ToxBench reports 0.057 for RF on Tox21)")
    print(f"\n  Model saved: {save_path}")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
