"""
data_preprocessing.py
----------------------
Loads the UCI "Default of Credit Card Clients" dataset, cleans it, runs a
short exploratory data analysis (EDA), and produces the train/test split
used by every downstream model.

Usage:
    python src/data_preprocessing.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import json

from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, FIGURES_DIR, REPORTS_DIR,
    TARGET_COL, ID_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    RANDOM_STATE, TEST_SIZE,
)

sns.set_style("whitegrid")


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw .xls file. The real header row is row 2 (index 1) --
    row 1 is a spreadsheet title ('X1','X2'...) inserted by the dataset donor."""
    df = pd.read_excel(path, engine="xlrd", header=1)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues documented in the UCI dataset card:

    - EDUCATION should be 1-4 but contains undocumented 0, 5, 6 values -> grouped into 'Other' (4)
    - MARRIAGE should be 1-3 but contains an undocumented 0 value -> grouped into 'Other' (3)
    - Duplicate rows / missing values are checked and reported
    """
    df = df.copy()

    # Drop the ID column - it carries no predictive signal
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # EDUCATION: valid = 1 (grad school), 2 (university), 3 (high school), 4 (other)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: valid = 1 (married), 2 (single), 3 (other)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes:
        df = df.drop_duplicates()

    # Missing values (dataset is documented as having none, but check defensively)
    missing = df.isnull().sum().sum()

    print(f"[clean_data] Removed {n_dupes} duplicate rows. Missing values remaining: {missing}.")
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Generate the small set of figures typically needed for the report's
    Literature Review / Methodology sections (class balance, correlations,
    key risk-driver distributions)."""

    # 1) Class balance
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x=TARGET_COL, data=df, hue=TARGET_COL, palette="Set2", legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Default (0)", "Default (1)"])
    plt.title("Class Distribution: Default Payment Next Month")
    plt.ylabel("Number of Clients")
    plt.xlabel("")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=200)
    plt.close()

    # 2) Correlation heatmap (numeric features + target)
    plt.figure(figsize=(12, 10))
    corr = df[NUMERIC_FEATURES + [TARGET_COL]].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Correlation Heatmap - Numeric Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_correlation_heatmap.png", dpi=200)
    plt.close()

    # 3) Repayment status (PAY_0, most recent month) vs default rate
    plt.figure(figsize=(7, 4))
    rate_by_pay0 = df.groupby("PAY_0")[TARGET_COL].mean().sort_index()
    rate_by_pay0.plot(kind="bar", color="#4C72B0")
    plt.title("Default Rate by Most Recent Repayment Status (PAY_0)")
    plt.xlabel("PAY_0 (-1=pay duly, 1+=months delayed)")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_default_rate_by_pay0.png", dpi=200)
    plt.close()

    # 4) Age distribution split by default status
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x="AGE", hue=TARGET_COL, fill=True, common_norm=False, alpha=0.4)
    plt.title("Age Distribution by Default Status")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_age_distribution.png", dpi=200)
    plt.close()

    # 5) Credit limit (LIMIT_BAL) distribution by default status
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x=TARGET_COL, y="LIMIT_BAL", hue=TARGET_COL,
                palette="Set2", legend=False)
    plt.xticks([0, 1], ["No Default (0)", "Default (1)"])
    plt.title("Credit Limit (LIMIT_BAL) Distribution by Default Status")
    plt.xlabel("")
    plt.ylabel("Credit Limit (NT$)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_05_limit_bal_by_default.png", dpi=200)
    plt.close()

    # 6) Default rate by education level
    plt.figure(figsize=(7, 4))
    edu_labels = {1: "Graduate\nSchool", 2: "University", 3: "High\nSchool", 4: "Other"}
    rate_by_edu = df.groupby("EDUCATION")[TARGET_COL].mean().sort_index()
    rate_by_edu.index = [edu_labels.get(i, str(i)) for i in rate_by_edu.index]
    rate_by_edu.plot(kind="bar", color="#DD8452", rot=0)
    plt.title("Default Rate by Education Level")
    plt.ylabel("Default Rate")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_06_default_rate_by_education.png", dpi=200)
    plt.close()

    print(f"[run_eda] Saved 6 EDA figures to {FIGURES_DIR}")


def save_eda_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Save the numeric facts behind the EDA -- consumed by the Streamlit
    dashboard's Exploratory Data Analysis tab and useful directly in the
    report's Methodology / Data section."""

    # Duplicates are only meaningful once the (unique) ID column is excluded --
    # matches the check performed inside clean_data().
    n_dupes = df_raw.drop(columns=[ID_COL]).duplicated().sum()
    class_counts = df_clean[TARGET_COL].value_counts().sort_index()
    class_pct = (class_counts / len(df_clean) * 100).round(2)

    corr_with_target = (
        df_clean[NUMERIC_FEATURES + [TARGET_COL]].corr()[TARGET_COL]
        .drop(TARGET_COL).sort_values(key=abs, ascending=False)
    )

    summary = {
        "raw_rows": int(df_raw.shape[0]),
        "raw_columns": int(df_raw.shape[1]),
        "duplicates_removed": int(n_dupes),
        "missing_values": int(df_clean.isnull().sum().sum()),
        "clean_rows": int(df_clean.shape[0]),
        "clean_columns": int(df_clean.shape[1]),
        "class_counts": {"No Default (0)": int(class_counts.get(0, 0)),
                          "Default (1)": int(class_counts.get(1, 0))},
        "class_pct": {"No Default (0)": float(class_pct.get(0, 0)),
                       "Default (1)": float(class_pct.get(1, 0))},
        "top5_correlated_with_target": {
            k: round(float(v), 4) for k, v in corr_with_target.head(5).items()
        },
    }

    with open(REPORTS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Descriptive statistics table for numeric features (min/mean/max/std etc.)
    df_clean[NUMERIC_FEATURES].describe().T.round(2).to_csv(REPORTS_DIR / "eda_descriptive_stats.csv")

    print(f"[save_eda_summary] Saved EDA summary -> {REPORTS_DIR / 'eda_summary.json'}")
    print(f"[save_eda_summary] Saved descriptive stats table -> {REPORTS_DIR / 'eda_descriptive_stats.csv'}")


def split_and_save(df: pd.DataFrame):
    """Stratified train/test split (stratified because the target is imbalanced
    ~78% / 22%) and persist the cleaned full dataset for reproducibility."""

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[split_and_save] Cleaned dataset saved to {PROCESSED_DATA_PATH} "
          f"({df.shape[0]} rows, {df.shape[1]} columns)")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[split_and_save] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"[split_and_save] Train default rate: {y_train.mean():.3f} | "
          f"Test default rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def main():
    print("=" * 70)
    print("STEP 1: DATA PREPROCESSING & EDA")
    print("=" * 70)
    df_raw = load_raw_data()
    print(f"[main] Loaded raw data: {df_raw.shape}")

    df_clean = clean_data(df_raw)
    run_eda(df_clean)
    save_eda_summary(df_raw, df_clean)
    split_and_save(df_clean)
    print("[main] Done.\n")


if __name__ == "__main__":
    main()
