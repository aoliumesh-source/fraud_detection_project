"""
risk_scoring.py
----------------
Builds the prototype risk-scoring system described in the proposal's
Methodology section: converts the best model's predicted probability of
default into Low / Medium / High risk categories, producing a scored
customer table and a supporting visualisation for the report and the
Streamlit dashboard.

Usage:
    python src/risk_scoring.py
"""

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RISK_THRESHOLDS


def assign_risk_band(prob: float) -> str:
    if prob < RISK_THRESHOLDS["low_max"]:
        return "Low Risk"
    elif prob < RISK_THRESHOLDS["medium_max"]:
        return "Medium Risk"
    else:
        return "High Risk"


def main():
    print("=" * 70)
    print("STEP 4: RISK SCORING DASHBOARD DATA")
    print("=" * 70)

    pipe = joblib.load(MODELS_DIR / "best_model.joblib")
    X_test = pd.read_csv(REPORTS_DIR / "X_test.csv")
    y_test = pd.read_csv(REPORTS_DIR / "y_test.csv").iloc[:, 0]

    probabilities = pipe.predict_proba(X_test)[:, 1]

    scored = X_test.copy()
    scored["Actual_Default"] = y_test.values
    scored["Predicted_Default_Probability"] = probabilities.round(4)
    scored["Risk_Category"] = [assign_risk_band(p) for p in probabilities]

    scored = scored.sort_values("Predicted_Default_Probability", ascending=False).reset_index(drop=True)
    scored.to_csv(REPORTS_DIR / "risk_scored_customers.csv", index=False)

    # Summary table: counts + actual default rate per risk band (validates the banding)
    summary = scored.groupby("Risk_Category").agg(
        Num_Customers=("Actual_Default", "count"),
        Actual_Default_Rate=("Actual_Default", "mean"),
        Avg_Predicted_Probability=("Predicted_Default_Probability", "mean"),
    ).reindex(["Low Risk", "Medium Risk", "High Risk"])
    summary.to_csv(REPORTS_DIR / "risk_band_summary.csv")

    print(summary.to_string())

    # Visualisation 1: customer counts per risk band
    plt.figure(figsize=(6, 4))
    order = ["Low Risk", "Medium Risk", "High Risk"]
    colors = {"Low Risk": "#4CAF50", "Medium Risk": "#FFC107", "High Risk": "#F44336"}
    sns.countplot(x="Risk_Category", data=scored, order=order,
                  hue="Risk_Category", palette=colors, legend=False)
    plt.title("Customer Distribution Across Risk Bands")
    plt.xlabel("")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_risk_band_distribution.png", dpi=200)
    plt.close()

    # Visualisation 2: actual default rate validates the banding is meaningful
    plt.figure(figsize=(6, 4))
    sns.barplot(x=summary.index, y="Actual_Default_Rate", data=summary.reset_index(),
                hue=summary.index, palette=colors, legend=False, order=order)
    plt.title("Actual Default Rate by Predicted Risk Band")
    plt.ylabel("Actual Default Rate")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_actual_default_rate_by_band.png", dpi=200)
    plt.close()

    print(f"\n[risk_scoring] Saved scored customer table -> "
          f"{REPORTS_DIR / 'risk_scored_customers.csv'}")
    print(f"[risk_scoring] Saved risk band summary -> {REPORTS_DIR / 'risk_band_summary.csv'}")
    print(f"[risk_scoring] Saved 2 figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
