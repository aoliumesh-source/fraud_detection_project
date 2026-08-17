"""
train_models.py
----------------
Trains and compares four classifiers on the credit default (fraud-risk proxy)
dataset, as specified in the project methodology:

    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. XGBoost

Each model is wrapped in a single sklearn Pipeline (preprocessing + estimator)
so that the exact same transformations are applied at training, testing, and
inference (Streamlit app) time -- this avoids train/serve skew.

Metrics reported: Accuracy, Precision, Recall, F1-score, ROC-AUC (as specified
in the project proposal, Section 4).

Usage:
    python src/train_models.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from xgboost import XGBClassifier

from config import (
    PROCESSED_DATA_PATH, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    TARGET_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    RANDOM_STATE, TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """Numeric features are scaled (needed for Logistic Regression); nominal
    categoricals are one-hot encoded. Tree-based models tolerate this fine too,
    so one shared preprocessor keeps the four pipelines directly comparable."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
        ]
    )


def get_model_zoo() -> dict:
    """Return the four candidate models specified in the methodology.
    class_weight / scale_pos_weight handle the ~78/22 class imbalance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE,
            scale_pos_weight=3.5,  # ~ (1 - default_rate) / default_rate
            n_jobs=-1,
        ),
    }


def evaluate_model(name, pipe, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"cm_{safe_name}.png", dpi=200)
    plt.close()

    # Save classification report as text
    report_txt = classification_report(y_test, y_pred, target_names=["No Default", "Default"])
    with open(REPORTS_DIR / f"classification_report_{safe_name}.txt", "w") as f:
        f.write(f"Classification Report - {name}\n")
        f.write("=" * 50 + "\n")
        f.write(report_txt)

    return metrics, y_proba


def plot_roc_curves(y_test, proba_dict: dict):
    plt.figure(figsize=(6, 5))
    for name, y_proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison Across Models")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_roc_curve_comparison.png", dpi=200)
    plt.close()


def plot_metric_comparison(results_df: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = results_df.set_index("Model")[metrics]
    ax = plot_df.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_metric_comparison.png", dpi=200)
    plt.close()


def main():
    print("=" * 70)
    print("STEP 2: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    models = get_model_zoo()

    results = []
    proba_dict = {}
    fitted_pipelines = {}

    for name, estimator in models.items():
        print(f"\n[train] Training {name} ...")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
        pipe.fit(X_train, y_train)

        metrics, y_proba = evaluate_model(name, pipe, X_test, y_test)
        results.append(metrics)
        proba_dict[name] = y_proba
        fitted_pipelines[name] = pipe

        print(f"[train] {name}: Accuracy={metrics['Accuracy']:.3f}  "
              f"Precision={metrics['Precision']:.3f}  Recall={metrics['Recall']:.3f}  "
              f"F1={metrics['F1-Score']:.3f}  ROC-AUC={metrics['ROC-AUC']:.3f}")

        joblib.dump(pipe, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    results_df.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)

    plot_roc_curves(y_test, proba_dict)
    plot_metric_comparison(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    joblib.dump(fitted_pipelines[best_model_name], MODELS_DIR / "best_model.joblib")

    with open(MODELS_DIR / "best_model_info.json", "w") as f:
        json.dump({"best_model": best_model_name,
                    "metrics": results_df.iloc[0].to_dict()}, f, indent=2)

    # Also persist test split for downstream SHAP / risk scoring scripts
    X_test.to_csv(REPORTS_DIR / "X_test.csv", index=False)
    y_test.to_csv(REPORTS_DIR / "y_test.csv", index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\n[main] Best model by ROC-AUC: {best_model_name}")
    print(f"[main] Saved: model comparison table, ROC curve, confusion matrices, "
          f"classification reports -> {REPORTS_DIR} / {FIGURES_DIR}")
    print(f"[main] Saved trained pipelines (incl. best_model.joblib) -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
