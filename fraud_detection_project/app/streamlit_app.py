"""
streamlit_app.py
-----------------
Interactive web application (the project's core artefact) that lets a user
enter a customer's profile and get:
  1. A real-time default/fraud-risk prediction from the best trained model
  2. A Low / Medium / High risk category
  3. A SHAP-based explanation of which factors drove that specific prediction
  4. A portfolio-level view of the risk-scored test set

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Allow importing config.py from ../src when run as `streamlit run app/streamlit_app.py`
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from config import (
    MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_PATH, TARGET_COL,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, RISK_THRESHOLDS,
)
from shap_utils import normalize_shap_output, patch_shap_xgboost_base_score_bug

patch_shap_xgboost_base_score_bug()

FIGURES_DIR_PATH = Path(__file__).resolve().parent.parent / "outputs" / "figures"

st.set_page_config(
    page_title="AI-Driven Fraud/Default Risk Detection",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_model():
    pipe = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "best_model_info.json") as f:
        info = json.load(f)
    return pipe, info


@st.cache_data
def load_scored_customers():
    return pd.read_csv(REPORTS_DIR / "risk_scored_customers.csv")


@st.cache_data
def load_metrics_table():
    return pd.read_csv(REPORTS_DIR / "model_comparison_metrics.csv")


@st.cache_data
def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_data
def load_eda_summary():
    with open(REPORTS_DIR / "eda_summary.json") as f:
        return json.load(f)


@st.cache_data
def load_descriptive_stats():
    return pd.read_csv(REPORTS_DIR / "eda_descriptive_stats.csv").rename(columns={"Unnamed: 0": "Feature"})


def risk_band(prob: float) -> str:
    if prob < RISK_THRESHOLDS["low_max"]:
        return "Low Risk"
    elif prob < RISK_THRESHOLDS["medium_max"]:
        return "Medium Risk"
    return "High Risk"


def risk_color(band: str) -> str:
    return {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}[band]


pipe, model_info = load_model()
best_model_name = model_info["best_model"]

st.title("💳 AI-Driven Fraud Risk Detection in Digital Payment Systems")
st.caption(
    "Prototype artefact for Module 7005SCN Individual Research Project — "
    "explainable ML-based credit default / fraud-risk scoring, built on the "
    "UCI Default of Credit Card Clients dataset."
)

tab_eda, tab1, tab2, tab3 = st.tabs(
    ["📈 Exploratory Data Analysis", "🔍 Predict a Customer",
     "📊 Model Performance", "📁 Portfolio Risk Overview"]
)

# ---------------------------------------------------------------------------
# TAB EDA - exploratory data analysis
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("Dataset Overview")
    df_processed = load_processed_data()
    eda = load_eda_summary()

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Raw Rows", f"{eda['raw_rows']:,}")
    o2.metric("Duplicates Removed", eda["duplicates_removed"])
    o3.metric("Missing Values", eda["missing_values"])
    o4.metric("Rows After Cleaning", f"{eda['clean_rows']:,}")

    st.markdown(
        f"The raw dataset contains **{eda['raw_rows']:,} clients** across "
        f"**{eda['raw_columns']} columns**. After removing "
        f"**{eda['duplicates_removed']} duplicate records** (no missing values "
        f"were found), **{eda['clean_rows']:,} clients** remain for analysis. "
        f"`EDUCATION` and `MARRIAGE` undocumented category codes (0, 5, 6) were "
        f"consolidated into an 'Other' category per the dataset's own "
        f"documentation before analysis."
    )

    st.markdown(
        f"**Class balance:** {eda['class_counts']['No Default (0)']:,} clients "
        f"did not default ({eda['class_pct']['No Default (0)']}%) versus "
        f"{eda['class_counts']['Default (1)']:,} who did "
        f"({eda['class_pct']['Default (1)']}%). This ~78/22 split is a "
        f"**moderate class imbalance**, which is why all four models below use "
        f"`class_weight='balanced'` / `scale_pos_weight`, and why ROC-AUC "
        f"(rather than accuracy alone) is used as the primary comparison metric."
    )

    with st.expander("📋 Descriptive statistics (numeric features)"):
        st.dataframe(load_descriptive_stats(), use_container_width=True)

    with st.expander("🔗 Top 5 features correlated with default (Pearson r)"):
        corr_df = pd.DataFrame(
            list(eda["top5_correlated_with_target"].items()),
            columns=["Feature", "Correlation with Default"]
        )
        st.dataframe(corr_df, use_container_width=True)
        st.caption(
            "All top predictors are repayment-status variables (PAY_0-PAY_5), "
            "confirming that recent repayment behaviour is a far stronger "
            "signal of default risk than demographic attributes."
        )

    st.divider()
    st.subheader("Visual Analysis")

    eda_plots = [
        ("01_class_distribution.png",
         "1. Class Distribution",
         "The target variable is imbalanced: roughly 3 in 4 clients did not "
         "default. This motivates the balanced class weighting used during "
         "model training and the choice of ROC-AUC / F1 over raw accuracy."),
        ("02_correlation_heatmap.png",
         "2. Correlation Heatmap (Numeric Features)",
         "The six PAY_* repayment-status variables are strongly correlated "
         "with each other (clients who fall behind tend to stay behind), and "
         "the BILL_AMT* variables are highly correlated with one another "
         "month-to-month. This informed feature scaling and later fed into "
         "the SHAP interpretation of which variables actually drive "
         "predictions."),
        ("03_default_rate_by_pay0.png",
         "3. Default Rate by Most Recent Repayment Status (PAY_0)",
         "Default rate rises sharply once a client is delayed by even one "
         "month (PAY_0 >= 1), and continues climbing with longer delays. "
         "This is the single strongest behavioural signal in the dataset, "
         "later confirmed as the top SHAP predictor."),
        ("04_age_distribution.png",
         "4. Age Distribution by Default Status",
         "Defaulting and non-defaulting clients show broadly similar age "
         "distributions, both peaking in the mid-to-late 20s/30s, indicating "
         "age alone is a weak standalone predictor relative to repayment "
         "history."),
        ("eda_05_limit_bal_by_default.png",
         "5. Credit Limit Distribution by Default Status",
         "Clients who default tend to have been extended lower credit limits "
         "on average, with defaulters' limits skewed toward the lower end "
         "of the range and fewer high-limit outliers than non-defaulters."),
        ("eda_06_default_rate_by_education.png",
         "6. Default Rate by Education Level",
         "Default rate varies moderately by education level, with 'Other' "
         "and 'High School' categories showing somewhat higher default rates "
         "than 'Graduate School' and 'University' -- a weaker but still "
         "relevant risk factor worth including alongside the payment-history "
         "features."),
    ]

    for i in range(0, len(eda_plots), 2):
        cols = st.columns(2)
        for col, (fname, title, analysis) in zip(cols, eda_plots[i:i + 2]):
            fig_path = FIGURES_DIR_PATH / fname
            with col:
                st.markdown(f"**{title}**")
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fname}")
                st.caption(analysis)

# ---------------------------------------------------------------------------
# TAB 1 - single customer prediction + SHAP explanation
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Enter Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL, NT$)", 10000, 1000000, 200000, step=10000)
        sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox(
            "Education", options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Other"}[x]
        )
        marriage = st.selectbox(
            "Marital Status", options=[1, 2, 3],
            format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x]
        )
        age = st.number_input("Age", 18, 90, 35)

    with col2:
        st.markdown("**Repayment status (last 6 months)**")
        st.caption("-1 = paid duly, 1+ = months delayed")
        pay_0 = st.slider("PAY_0 (most recent)", -2, 9, 0)
        pay_2 = st.slider("PAY_2", -2, 9, 0)
        pay_3 = st.slider("PAY_3", -2, 9, 0)
        pay_4 = st.slider("PAY_4", -2, 9, 0)
        pay_5 = st.slider("PAY_5", -2, 9, 0)
        pay_6 = st.slider("PAY_6", -2, 9, 0)

    with col3:
        st.markdown("**Bill & payment amounts (NT$, last month)**")
        bill_amt1 = st.number_input("Bill Amount (BILL_AMT1)", -50000, 1000000, 50000, step=1000)
        pay_amt1 = st.number_input("Payment Amount (PAY_AMT1)", 0, 1000000, 2000, step=500)
        # Remaining months default to the same values for simplicity in the demo UI
        bill_amt2 = bill_amt3 = bill_amt4 = bill_amt5 = bill_amt6 = bill_amt1
        pay_amt2 = pay_amt3 = pay_amt4 = pay_amt5 = pay_amt6 = pay_amt1

    if st.button("🔮 Predict Risk", type="primary"):
        customer = pd.DataFrame([{
            "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage,
            "AGE": age, "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4,
            "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": bill_amt1, "BILL_AMT2": bill_amt2, "BILL_AMT3": bill_amt3,
            "BILL_AMT4": bill_amt4, "BILL_AMT5": bill_amt5, "BILL_AMT6": bill_amt6,
            "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
            "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6,
        }])

        prob = pipe.predict_proba(customer)[0, 1]
        band = risk_band(prob)

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Default Probability", f"{prob:.1%}")
        r2.metric("Risk Category", f"{risk_color(band)} {band}")
        r3.metric("Model Used", best_model_name)

        # Local SHAP explanation for this customer
        st.markdown("#### Why this prediction? (SHAP explanation)")
        preprocessor = pipe.named_steps["preprocessor"]
        classifier = pipe.named_steps["classifier"]
        X_t = preprocessor.transform(customer)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        num_names = NUMERIC_FEATURES
        cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = num_names + cat_names
        X_t_df = pd.DataFrame(X_t, columns=feature_names)

        try:
            explainer = shap.TreeExplainer(classifier)
            raw_shap_values = explainer.shap_values(X_t_df)
            shap_vals, _ = normalize_shap_output(raw_shap_values, explainer.expected_value)
            contrib = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_vals[0],
            }).sort_values("SHAP Value", key=abs, ascending=False).head(8)

            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#F44336" if v > 0 else "#4CAF50" for v in contrib["SHAP Value"]]
            ax.barh(contrib["Feature"], contrib["SHAP Value"], color=colors)
            ax.set_xlabel("SHAP value (→ pushes toward Default)")
            ax.invert_yaxis()
            st.pyplot(fig)
            st.caption("🔴 Red bars increase predicted default risk. 🟢 Green bars decrease it.")
        except Exception as e:
            st.info(f"SHAP explanation unavailable for this model type ({e}).")

# ---------------------------------------------------------------------------
# TAB 2 - model performance comparison
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Model Comparison (held-out test set)")
    metrics_df = load_metrics_table()
    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
            "F1-Score": "{:.3f}", "ROC-AUC": "{:.3f}",
        }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#d4f7d4"),
        use_container_width=True,
    )
    st.info(f"**Selected production model:** {best_model_name} "
            f"(highest ROC-AUC on held-out test data)")

    fig_col1, fig_col2 = st.columns(2)
    roc_path = Path(__file__).resolve().parent.parent / "outputs" / "figures" / "05_roc_curve_comparison.png"
    shap_path = Path(__file__).resolve().parent.parent / "outputs" / "figures" / "08_shap_feature_importance_bar.png"
    if roc_path.exists():
        fig_col1.image(str(roc_path), caption="ROC Curve Comparison")
    if shap_path.exists():
        fig_col2.image(str(shap_path), caption="Global Feature Importance (SHAP)")

# ---------------------------------------------------------------------------
# TAB 3 - portfolio-level risk overview
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Test-Set Portfolio Risk Overview")
    scored = load_scored_customers()

    band_counts = scored["Risk_Category"].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"]
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Low Risk", int(band_counts.get("Low Risk", 0)))
    c2.metric("🟡 Medium Risk", int(band_counts.get("Medium Risk", 0)))
    c3.metric("🔴 High Risk", int(band_counts.get("High Risk", 0)))

    st.markdown("#### Actual Default Rate by Predicted Risk Band")
    st.caption("Validates that the risk bands are meaningful: higher predicted risk should mean higher observed default rate.")
    summary_path = REPORTS_DIR / "risk_band_summary.csv"
    if summary_path.exists():
        st.dataframe(pd.read_csv(summary_path), use_container_width=True)

    st.markdown("#### Scored Customers (test set)")
    band_filter = st.multiselect(
        "Filter by risk category", options=["Low Risk", "Medium Risk", "High Risk"],
        default=["High Risk"]
    )
    display_df = scored[scored["Risk_Category"].isin(band_filter)] if band_filter else scored
    st.dataframe(display_df.head(200), use_container_width=True)
    st.caption(f"Showing {min(len(display_df), 200)} of {len(display_df)} matching rows.")

st.divider()
st.caption(
    "Dataset: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H"
)
