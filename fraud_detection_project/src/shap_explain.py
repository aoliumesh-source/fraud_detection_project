"""
shap_explain.py
----------------
Applies SHAP (SHapley Additive exPlanations) to the best-performing model to
satisfy the project's explainable-AI (XAI) objective: identifying which
variables contribute most to a "High Risk" prediction, and producing
figures/artifacts for the report's Artefact Design & Development section.

Usage:
    python src/shap_explain.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MODELS_DIR, FIGURES_DIR, REPORTS_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from shap_utils import normalize_shap_output, patch_shap_xgboost_base_score_bug

patch_shap_xgboost_base_score_bug()


def get_feature_names(preprocessor) -> list:
    """Recover human-readable feature names after ColumnTransformer transforms
    (StandardScaler keeps names; OneHotEncoder expands them)."""
    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names


def main():
    print("=" * 70)
    print("STEP 3: EXPLAINABLE AI (SHAP)")
    print("=" * 70)

    pipe = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "best_model_info.json") as f:
        info = json.load(f)
    model_name = info["best_model"]
    print(f"[shap] Explaining best model: {model_name}")

    preprocessor = pipe.named_steps["preprocessor"]
    classifier = pipe.named_steps["classifier"]

    X_test = pd.read_csv(REPORTS_DIR / "X_test.csv")
    # Use a sample for speed on non-tree models / large test sets
    sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    X_sample_transformed = preprocessor.transform(sample)
    if hasattr(X_sample_transformed, "toarray"):
        X_sample_transformed = X_sample_transformed.toarray()

    feature_names = get_feature_names(preprocessor)
    X_sample_df = pd.DataFrame(X_sample_transformed, columns=feature_names)

    # TreeExplainer works for Decision Tree / Random Forest / XGBoost.
    # Falls back to a general Explainer for other model types (e.g. Logistic Regression).
    # NOTE: shap.TreeExplainer.shap_values() returns DIFFERENT shapes depending on
    # the SHAP version and the model type -- a list of per-class arrays, a 3D
    # (n_samples, n_features, n_classes) array, or a plain 2D array. normalize_shap_output()
    # (src/shap_utils.py) collapses all of these into one consistent 2D array + scalar
    # expected value, for the positive ("Default") class.
    tree_based = model_name in ("Decision Tree", "Random Forest", "XGBoost")
    if tree_based:
        explainer = shap.TreeExplainer(classifier)
        raw_shap_values = explainer.shap_values(X_sample_df)
        shap_values, expected_value = normalize_shap_output(raw_shap_values, explainer.expected_value)
    else:
        explainer = shap.Explainer(classifier, X_sample_df)
        exp = explainer(X_sample_df)
        shap_values, expected_value = normalize_shap_output(exp.values, exp.base_values)

    # 1) Global summary (beeswarm) plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample_df, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Global feature importance (mean |SHAP value|) bar chart
    plt.figure()
    shap.summary_plot(shap_values, X_sample_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_shap_feature_importance_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Save ranked feature importance table (for the report's Evaluation section)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP_Value": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP_Value", ascending=False).reset_index(drop=True)
    importance_df.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    # 4) Single-customer local explanation (waterfall) for one high-risk example
    proba = pipe.predict_proba(sample)[:, 1]
    idx_high_risk = int(np.argmax(proba))
    plt.figure()
    expl = shap.Explanation(
        values=shap_values[idx_high_risk],
        base_values=expected_value,
        data=X_sample_df.iloc[idx_high_risk].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(expl, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_shap_waterfall_example_customer.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nTop 10 predictors of high default risk (mean |SHAP value|):")
    print(importance_df.head(10).to_string(index=False))
    print(f"\n[shap] Saved SHAP figures -> {FIGURES_DIR}")
    print(f"[shap] Saved feature importance table -> {REPORTS_DIR / 'shap_feature_importance.csv'}")


if __name__ == "__main__":
    main()
