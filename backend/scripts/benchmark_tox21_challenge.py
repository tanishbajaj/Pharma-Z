"""
Benchmark the toxicity model on the OFFICIAL Tox21 Challenge 2014 split.

This is the protocol used by the leaderboard in Ebner et al. 2025
(arXiv:2511.14744, "Measuring AI Progress in Drug Discovery: A Reproducible
Leaderboard for the Tox21 Challenge"): train on the 12,060 challenge training
compounds, evaluate on the 647-compound held-out final evaluation set, score
by per-endpoint AUC averaged across the 12 assays.

This is NOT the same as scripts/train_toxicity.py, which uses the smaller
MoleculeNet/DeepChem CSV (7,831 molecules) and our own scaffold split. Ebner
et al. state MoleculeNet-derived results are "not directly comparable to the
original leaderboard", so the two scripts answer different questions:

    train_toxicity.py            -> what ships in the pipeline (scaffold split)
    benchmark_tox21_challenge.py -> how it scores against published numbers

Data (Tox21 Machine Learning Data Set, DeepTox group, JKU Linz):
    http://bioinf.jku.at/research/DeepTox/tox21.html
    tox21.sdf.gz, tox21_labels_train.csv.gz, tox21_labels_test.csv.gz
Note: that host's HTTPS certificate has expired, so the files are served over
plain HTTP. They are public, unauthenticated scientific data.

Usage:
    python scripts/benchmark_tox21_challenge.py
    python scripts/benchmark_tox21_challenge.py --n-seeds 5
"""

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.config import settings
from app.models.toxicity_model import ToxicityPredictor, smiles_to_fingerprint

# The label CSVs use dots; our pipeline uses dashes.
ASSAYS = [
    "NR.AhR", "NR.AR", "NR.AR.LBD", "NR.Aromatase", "NR.ER", "NR.ER.LBD",
    "NR.PPAR.gamma", "SR.ARE", "SR.ATAD5", "SR.HSE", "SR.MMP", "SR.p53",
]

# Random Forest per-endpoint AUC on this exact split, Ebner et al. 2025 Table 1
# (mean over 5 seeds). Overall reported as median with MAD: 0.851 +/- 0.002.
EBNER_RF = {
    "NR.AR": 0.829, "NR.AR.LBD": 0.783, "NR.AhR": 0.768, "NR.Aromatase": 0.917,
    "NR.ER": 0.820, "NR.ER.LBD": 0.814, "NR.PPAR.gamma": 0.774, "SR.ARE": 0.827,
    "SR.ATAD5": 0.800, "SR.HSE": 0.811, "SR.MMP": 0.842, "SR.p53": 0.946,
}
EBNER_OVERALL = {
    "Random Forest": 0.851, "DeepTox (challenge winner)": 0.862, "SNN": 0.856,
    "TabPFN": 0.826, "XGBoost": 0.821, "Chemprop": 0.818, "GIN": 0.806,
}


def load_structures(sdf_path):
    """Map NCGC compound ID -> Morgan fingerprint."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    fingerprints = {}
    supplier = Chem.ForwardSDMolSupplier(gzip.open(sdf_path))
    for mol in supplier:
        if mol is None:
            continue
        try:
            identifier = mol.GetProp("_Name")
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            continue
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is not None:
            fingerprints[identifier] = fingerprint
    return fingerprints


def load_test_structures(tsv_path):
    """
    Map NCGC ID -> fingerprint for the 647 final-evaluation compounds.

    tox21.sdf.gz holds only the training structures, so the held-out test
    structures come from the NCATS challenge file
    (tox21_10k_challenge_scoresmiles), whose Sample IDs match the label CSV.
    """
    frame = pd.read_csv(tsv_path, sep="\t")
    frame.columns = ["smiles", "id"]
    fingerprints = {}
    for smiles, identifier in zip(frame["smiles"], frame["id"]):
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is not None:
            fingerprints[identifier] = fingerprint
    return fingerprints


def build_matrix(labels_frame, fingerprints):
    """Align label rows with available structures."""
    ids = [i for i in labels_frame.index if i in fingerprints]
    features = np.stack([fingerprints[i] for i in ids])
    labels = labels_frame.loc[ids, ASSAYS].values.astype(np.float32)
    return features, labels, ids


def load_dense(data_dir, train_labels, test_labels):
    """
    The 801 precomputed chemical descriptors shipped with the DeepTox dataset,
    already aligned to the official split. These are the features the published
    Random Forest baseline is built on — our pipeline uses 2048-bit Morgan
    fingerprints instead, so benchmarking both separates "different model" from
    "different representation".
    """
    train = pd.read_csv(os.path.join(data_dir, "tox21_dense_train.csv.gz"), index_col=0)
    test = pd.read_csv(os.path.join(data_dir, "tox21_dense_test.csv.gz"), index_col=0)

    train_ids = [i for i in train_labels.index if i in train.index]
    test_ids = [i for i in test_labels.index if i in test.index]

    x_train = np.nan_to_num(train.loc[train_ids].values.astype(np.float32))
    x_test = np.nan_to_num(test.loc[test_ids].values.astype(np.float32))
    y_train = train_labels.loc[train_ids, ASSAYS].values.astype(np.float32)
    y_test = test_labels.loc[test_ids, ASSAYS].values.astype(np.float32)
    return x_train, y_train, x_test, y_test


def benchmark(x_train, y_train, x_test, y_test, args, tag):
    """Train n_seeds masked RFs and score per-assay AUC on the held-out set."""
    print(f"\n{'=' * 72}")
    print(f"FEATURES: {tag}   train {x_train.shape} test {x_test.shape}")
    print(f"{'=' * 72}")

    per_seed_assay, per_seed_mean = [], []
    for offset in range(args.n_seeds):
        seed = args.seed + offset
        model = ToxicityPredictor(n_estimators=args.n_estimators, random_state=seed)
        start = time.time()
        model.train_masked(x_train, y_train, label_names=ASSAYS)
        probabilities = model.predict_proba(x_test)

        scores = {}
        for index, assay in enumerate(ASSAYS):
            valid = ~np.isnan(y_test[:, index])
            y_true = y_test[valid, index]
            if len(np.unique(y_true)) < 2:
                continue
            scores[assay] = roc_auc_score(y_true, probabilities[valid, index])
        per_seed_assay.append(scores)
        per_seed_mean.append(float(np.mean(list(scores.values()))))
        print(f"  seed {seed}: mean AUC {per_seed_mean[-1]:.4f}  ({time.time() - start:.0f}s)")

    print(f"\n  {'assay':<16}{'ours':>9}{'+/-':>8}{'Ebner RF':>11}{'delta':>9}")
    print(f"  {'-' * 54}")
    summary = {}
    for assay in ASSAYS:
        values = [s[assay] for s in per_seed_assay if assay in s]
        if not values:
            continue
        mean, std = float(np.mean(values)), float(np.std(values))
        reference = EBNER_RF[assay]
        summary[assay] = {"ours": round(mean, 4), "std": round(std, 4),
                          "ebner_rf": reference, "delta": round(mean - reference, 4)}
        print(f"  {assay:<16}{mean:>9.4f}{std:>8.4f}{reference:>11.3f}"
              f"{mean - reference:>+9.4f}")

    our_mean = float(np.mean(per_seed_mean))
    our_std = float(np.std(per_seed_mean))
    print(f"  {'-' * 54}")
    print(f"  {'MEAN':<16}{our_mean:>9.4f}{our_std:>8.4f}"
          f"{EBNER_OVERALL['Random Forest']:>11.3f}"
          f"{our_mean - EBNER_OVERALL['Random Forest']:>+9.4f}")

    return {
        "features": tag,
        "n_features": int(x_train.shape[1]),
        "mean_roc_auc": round(our_mean, 4),
        "std_roc_auc": round(our_std, 4),
        "per_seed_mean": [round(v, 4) for v in per_seed_mean],
        "per_assay": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Official Tox21 Challenge benchmark")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(settings.data_raw_dir, "tox21_challenge"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", choices=["morgan", "dense", "both"],
                        default="both",
                        help="morgan = our pipeline's representation; "
                             "dense = the 801 descriptors the published RF uses")
    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print("Tox21 Challenge 2014 — OFFICIAL SPLIT BENCHMARK")
    print(f"{'=' * 72}")

    sdf = os.path.join(args.data_dir, "tox21.sdf.gz")
    train_csv = os.path.join(args.data_dir, "tox21_labels_train.csv.gz")
    test_csv = os.path.join(args.data_dir, "tox21_labels_test.csv.gz")
    test_tsv = os.path.join(args.data_dir, "tox21_test_smiles.tsv")
    for path in (sdf, train_csv, test_csv, test_tsv):
        if not os.path.exists(path):
            print(f"  x Missing {path}")
            print(f"    Download from http://bioinf.jku.at/research/DeepTox/")
            sys.exit(1)

    train_labels = pd.read_csv(train_csv, index_col=0)
    test_labels = pd.read_csv(test_csv, index_col=0)
    print(f"\n  Official split: {len(train_labels)} train / {len(test_labels)} test")

    results = {}
    wanted = ["morgan", "dense"] if args.features == "both" else [args.features]

    for tag in wanted:
        if tag == "morgan":
            print("\n  Fingerprinting (Morgan r=2, 2048 bits) ...")
            start = time.time()
            train_fp = load_structures(sdf)
            test_fp = load_test_structures(
                os.path.join(args.data_dir, "tox21_test_smiles.tsv")
            )
            print(f"  {len(train_fp)} train / {len(test_fp)} test "
                  f"({time.time() - start:.0f}s)")
            x_train, y_train, _ = build_matrix(train_labels, train_fp)
            x_test, y_test, _ = build_matrix(test_labels, test_fp)
            label = "Morgan 2048 (what our pipeline uses)"
        else:
            x_train, y_train, x_test, y_test = load_dense(
                args.data_dir, train_labels, test_labels
            )
            label = "DeepTox 801 dense descriptors (published baseline)"

        measured = ~np.isnan(y_train)
        print(f"  measured train labels: {measured.sum()}/{measured.size} "
              f"({100 * measured.mean():.1f}%) — masked, not zero-filled")
        results[tag] = benchmark(x_train, y_train, x_test, y_test, args, label)

    print(f"\n{'=' * 72}")
    print("  OFFICIAL TOX21 CHALLENGE LEADERBOARD (Ebner et al. 2025)")
    print(f"{'=' * 72}")
    rows = [(name, value, False) for name, value in EBNER_OVERALL.items()]
    for tag, result in results.items():
        rows.append((f">> OURS ({tag})", result["mean_roc_auc"], True))
    for name, value, ours in sorted(rows, key=lambda r: -r[1]):
        print(f"    {name:<34} {value:.3f}{'   <-- us' if ours else ''}")

    out = os.path.join(settings.saved_models_dir, "toxicity",
                       "benchmark_tox21_challenge.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({
            "protocol": "Official Tox21 Challenge 2014 split",
            "train_compounds": int(len(train_labels)),
            "test_compounds": int(len(test_labels)),
            "n_seeds": args.n_seeds,
            "n_estimators": args.n_estimators,
            "missing_label_handling": "masked",
            "results_by_features": results,
            "reference_leaderboard": EBNER_OVERALL,
            "reference": "Ebner et al. 2025, arXiv:2511.14744, Table 1",
            "data_source": "http://bioinf.jku.at/research/DeepTox/ (DeepTox, JKU)",
        }, handle, indent=2)
    print(f"\n  Saved: {out}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
