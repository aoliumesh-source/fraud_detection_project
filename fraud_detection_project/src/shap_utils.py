"""
shap_utils.py
-------------
shap.TreeExplainer.shap_values() returns different shapes depending on the
SHAP library version AND the model type, which is a common source of bugs:

    - list of per-class arrays, e.g. [array(n_samples, n_features), array(...)]
      (older SHAP versions, some classifiers)
    - a single 3D array of shape (n_samples, n_features, n_classes)
      (newer SHAP versions, observed here for DecisionTreeClassifier and
      RandomForestClassifier)
    - a single 2D array of shape (n_samples, n_features)
      (observed here for XGBoost's binary:logistic objective)

`explainer.expected_value` similarly can be a scalar, a 2-element array, or
a list, depending on the same factors.

normalize_shap_output() converts any of these into a single, consistent
2D array (n_samples, n_features) for the positive ("Default"/high-risk)
class, plus a single scalar expected value -- so the rest of the code never
has to think about SHAP's output shape again.
"""

import numpy as np


def patch_shap_xgboost_base_score_bug() -> None:
    """
    Works around a real-world SHAP/XGBoost incompatibility observed across
    SHAP versions 0.44.x-0.49.x (and possibly others) when paired with
    XGBoost >= 2.0.

    XGBoost >= 2.0 internally represents `base_score` as a vector (to support
    multi-target models) and serialises it as a bracketed string, e.g.
    '[4.986033E-1]', even for an ordinary single-output binary classifier.
    Affected SHAP versions call `float(...)` directly on that string when
    loading an XGBoost model into `shap.TreeExplainer`, which raises:

        ValueError: could not convert string to float: '[4.986033E-1]'

    This cannot be fixed by editing the XGBoost model itself (its own
    `save_model`/`load_model` round-trip re-derives the same bracketed
    format), so this patches SHAP's internal UBJSON model decoder -- used
    only when loading XGBoost models -- to strip the brackets from any
    'base_score' field it finds before SHAP tries to parse it.

    Safe to call multiple times and safe on SHAP versions that don't have
    this bug (verified as a no-op there): it only rewrites a string that
    starts with '[', which a fixed/patched SHAP version wouldn't produce
    in the first place. Call this once, before creating any
    shap.TreeExplainer for an XGBoost model.
    """
    try:
        from shap.explainers import _tree as _shap_tree_module
    except ImportError:
        return  # SHAP isn't installed / layout differs -- nothing to patch

    original_decode = getattr(_shap_tree_module, "decode_ubjson_buffer", None)
    if original_decode is None or getattr(original_decode, "_base_score_patch_applied", False):
        return  # nothing to patch, or already patched in this process

    def _clean_base_score(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "base_score" and isinstance(value, str) and value.strip().startswith("["):
                    obj[key] = value.strip("[]")
                else:
                    _clean_base_score(value)
        elif isinstance(obj, list):
            for item in obj:
                _clean_base_score(item)
        return obj

    def patched_decode_ubjson_buffer(fd):
        result = original_decode(fd)
        return _clean_base_score(result)

    patched_decode_ubjson_buffer._base_score_patch_applied = True
    _shap_tree_module.decode_ubjson_buffer = patched_decode_ubjson_buffer


def normalize_shap_output(raw_shap_values, expected_value, positive_class_index: int = 1):
    """Return (shap_values_2d, expected_value_scalar) for the positive class.

    Works regardless of whether raw_shap_values is a list of per-class
    arrays, a 3D (n_samples, n_features, n_classes) array, or already a
    plain 2D (n_samples, n_features) array.
    """

    # Case 1: list of per-class arrays
    if isinstance(raw_shap_values, list):
        idx = positive_class_index if len(raw_shap_values) > positive_class_index else 0
        shap_values = np.asarray(raw_shap_values[idx])
        ev_arr = np.atleast_1d(expected_value)
        ev = ev_arr[idx] if ev_arr.shape[0] > idx else ev_arr[0]
        return shap_values, float(ev)

    raw_shap_values = np.asarray(raw_shap_values)

    # Case 2: 3D array (n_samples, n_features, n_classes)
    if raw_shap_values.ndim == 3:
        n_classes = raw_shap_values.shape[-1]
        idx = positive_class_index if n_classes > positive_class_index else n_classes - 1
        shap_values = raw_shap_values[:, :, idx]
        ev_arr = np.atleast_1d(expected_value)
        ev = ev_arr[idx] if ev_arr.shape[0] > idx else ev_arr[0]
        return shap_values, float(ev)

    # Case 3: already 2D (n_samples, n_features) -- e.g. XGBoost binary objective
    ev_arr = np.atleast_1d(expected_value)
    ev = ev_arr[-1] if ev_arr.shape[0] > 1 else ev_arr[0]
    return raw_shap_values, float(ev)
