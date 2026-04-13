"""
SHAP Explainer

Explainability module using SHAP (SHapley Additive exPlanations)
to interpret model predictions for drug discovery.
"""

from typing import Any, Optional, List, Dict

import numpy as np

from app.utils.logging_config import logger

# SHAP is an optional dependency
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — explainability features unavailable")


def explain_prediction(
    model: Any,
    input_data: np.ndarray,
    feature_names: List[str] = None,
    num_background_samples: int = 100,
    background_data: np.ndarray = None,
) -> dict:
    """
    Generate SHAP explanation for a model prediction.

    Args:
        model:                  Trained sklearn or callable model.
        input_data:             Input sample(s) to explain, shape (1, n_features) or (n, n_features).
        feature_names:          Names of input features.
        num_background_samples: Number of background samples for SHAP.
        background_data:        Background dataset (uses synthetic if None).

    Returns:
        Dict with shap_values, base_value, feature_importance.
    """
    if not SHAP_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "SHAP is not installed",
            "shap_values": [],
            "feature_importance": {},
        }

    try:
        if background_data is None:
            background_data = np.zeros((num_background_samples, input_data.shape[-1]))

        # Choose appropriate explainer
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, background_data)
        elif hasattr(model, "predict"):
            explainer = shap.Explainer(model.predict, background_data)
        else:
            explainer = shap.Explainer(model, background_data)

        shap_values = explainer(input_data)

        # Extract feature importance
        if hasattr(shap_values, "values"):
            values = shap_values.values
            if values.ndim == 3:
                # Multi-class: take mean absolute across classes
                importance = np.mean(np.abs(values), axis=(0, 2))
            else:
                importance = np.mean(np.abs(values), axis=0)
        else:
            importance = np.zeros(input_data.shape[-1])

        # Build feature importance dict
        feature_importance = {}
        for i, imp in enumerate(importance):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
            feature_importance[name] = round(float(imp), 6)

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        result = {
            "status": "success",
            "shap_values": values.tolist() if hasattr(values, "tolist") else [],
            "base_value": float(shap_values.base_values.mean()) if hasattr(shap_values, "base_values") else 0.0,
            "feature_importance": feature_importance,
            "top_features": list(feature_importance.keys())[:10],
        }

        logger.info(f"SHAP explanation generated: {len(feature_importance)} features")
        return result

    except Exception as error:
        logger.error(f"SHAP explanation failed: {error}")
        return {
            "status": "failed",
            "error": str(error),
            "shap_values": [],
            "feature_importance": {},
        }
