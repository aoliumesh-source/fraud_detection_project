"""
config.py
---------
Central configuration for the AI-Driven Fraud/Default Risk Detection
Framework (Module 7005SCN Individual Research Project).

Keeping paths and constants in one place means every script (preprocessing,
training, SHAP explanation, risk scoring, Streamlit app) stays in sync.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "default_of_credit_card_clients.xls"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed_data.csv"

FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

for _dir in (FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
TARGET_COL = "default payment next month"   # 1 = high risk / default, 0 = low risk
ID_COL = "ID"

# Columns that are genuinely continuous / ordinal-numeric and benefit from scaling
NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# Nominal categorical columns (one-hot encoded)
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Risk scoring thresholds on predicted probability of default (0-1)
RISK_THRESHOLDS = {
    "low_max": 0.30,      # prob < 0.30  -> Low Risk
    "medium_max": 0.60,   # 0.30 <= prob < 0.60 -> Medium Risk
    # prob >= 0.60 -> High Risk
}
