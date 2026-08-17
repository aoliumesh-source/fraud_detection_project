"""
main.py
-------
Runs the complete pipeline end to end, in the order described in the
project proposal's Methodology (Section 4):

    1. Data preprocessing & EDA
    2. Model training & evaluation (Logistic Regression, Decision Tree,
       Random Forest, XGBoost)
    3. SHAP explainability
    4. Risk scoring (Low / Medium / High)

After this finishes, launch the interactive dashboard separately with:
    streamlit run app/streamlit_app.py

Usage:
    python src/main.py
"""

import time
import data_preprocessing
import train_models
import shap_explain
import risk_scoring


def main():
    start = time.time()
    data_preprocessing.main()
    train_models.main()
    shap_explain.main()
    risk_scoring.main()
    elapsed = time.time() - start

    print("=" * 70)
    print(f"PIPELINE COMPLETE in {elapsed:.1f} seconds")
    print("=" * 70)
    print("Next step: launch the interactive dashboard with:")
    print("    streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
