"""
Binding Affinity Evaluation Metrics

Implements domain-standard metrics for drug-target affinity prediction:
- Concordance Index (CI) — primary ranking metric
- RMSE — Root Mean Squared Error
- MSE — Mean Squared Error (training loss)
- Pearson r — correlation coefficient

Reference: Öztürk et al., "DeepDTA: deep drug-target binding affinity prediction"
"""

import numpy as np


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Concordance Index (CI) for binding affinity predictions.

    CI measures whether predicted binding affinities preserve the 
    pairwise ordering of true affinities. A CI of 0.5 = random, 1.0 = perfect.

    Formula (from DeepDTA paper):
        CI = (1/Z) * Σ h(p_i - p_j) for all pairs where y_i > y_j
        h(x) = 1 if x > 0, 0.5 if x == 0, 0 if x < 0

    This is O(N²) — for KIBA test sets (~23K), takes ~2 seconds.

    Args:
        y_true: Array of true affinity values.
        y_pred: Array of predicted affinity values.

    Returns:
        Concordance index in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    assert len(y_true) == len(y_pred), \
        f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"

    n = len(y_true)
    if n < 2:
        return 0.0

    concordant = 0.0
    total_comparable = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue  # Skip ties in ground truth

            total_comparable += 1.0

            if y_true[i] > y_true[j]:
                if y_pred[i] > y_pred[j]:
                    concordant += 1.0
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
            else:
                if y_pred[i] < y_pred[j]:
                    concordant += 1.0
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5

    if total_comparable == 0:
        return 0.0

    return concordant / total_comparable


def concordance_index_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Vectorized (faster) CI computation using NumPy broadcasting.

    Same result as concordance_index() but ~10-50x faster on large arrays.
    Uses more memory: O(N²).

    Args:
        y_true: Array of true affinity values.
        y_pred: Array of predicted affinity values.

    Returns:
        Concordance index in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    n = len(y_true)
    if n < 2:
        return 0.0

    # For very large arrays, fall back to loop version to avoid OOM
    if n > 50000:
        return concordance_index(y_true, y_pred)

    # Pairwise differences (upper triangle)
    true_diff = y_true[:, None] - y_true[None, :]   # (N, N)
    pred_diff = y_pred[:, None] - y_pred[None, :]    # (N, N)

    # Only consider pairs where true values differ (upper triangle)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1) & (true_diff != 0)

    # For pairs where y_true[i] > y_true[j]:
    #   concordant if y_pred[i] > y_pred[j]
    #   half concordant if y_pred[i] == y_pred[j]
    # Flip sign: ensure true_diff > 0 for the counted direction
    td = true_diff[mask]
    pd_ = pred_diff[mask]

    # Make sure we count in same direction
    sign_match = np.sign(td) == np.sign(pd_)
    pred_tie = (pd_ == 0)

    concordant = np.sum(sign_match) + 0.5 * np.sum(pred_tie & ~sign_match)
    total = len(td)

    if total == 0:
        return 0.0

    return float(concordant / total)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values.

    Returns:
        MSE value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient.

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values.

    Returns:
        Pearson r in [-1, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all binding affinity metrics at once.

    Args:
        y_true: Array of true affinity values.
        y_pred: Array of predicted affinity values.

    Returns:
        Dict with keys: mse, rmse, ci, pearson_r
    """
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "ci": concordance_index_fast(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
    }
