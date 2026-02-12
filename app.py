import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    layout="wide"
)

st.title("🩺 Breast Cancer Classification — ML Model Comparison")

st.write("""
Upload **test CSV data** and evaluate predictions using
six trained machine-learning models.

Dataset must contain:
• same 30 feature columns  
• diagnosis column for evaluation
""")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


# ============================================================
# LOAD MODELS + SCALER
# ============================================================

@st.cache_resource
def load_assets():

    models = {
        "Logistic Regression": joblib.load(
            os.path.join(MODEL_DIR, "logistic.pkl")
        ),
        "Decision Tree": joblib.load(
            os.path.join(MODEL_DIR, "decision_tree.pkl")
        ),
        "KNN": joblib.load(
            os.path.join(MODEL_DIR, "knn.pkl")
        ),
        "Naive Bayes": joblib.load(
            os.path.join(MODEL_DIR, "naive_bayes.pkl")
        ),
        "Random Forest": joblib.load(
            os.path.join(MODEL_DIR, "random_forest.pkl")
        ),
        "XGBoost": joblib.load(
            os.path.join(MODEL_DIR, "xgboost.pkl")
        ),
    }

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    return models, scaler


models, scaler = load_assets()


# ============================================================
# SIDEBAR — MODEL SELECTION
# ============================================================

st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose classification model:",
    list(models.keys())
)

model = models[model_name]


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CSV file (test data only)",
    type=["csv"]
)


# ============================================================
# MAIN INFERENCE LOGIC
# ============================================================

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Drop unwanted columns
    df.drop(columns=["id"], inplace=True, errors="ignore")
    df.drop(columns=["Unnamed: 32"], inplace=True, errors="ignore")

    if "diagnosis" not in df.columns:
        st.error("Uploaded CSV must contain 'diagnosis' column.")
        st.stop()

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"].map({"M": 1, "B": 0}).fillna(df["diagnosis"])

    # Scaling if required
    scaled_models = ["Logistic Regression", "KNN", "Naive Bayes"]

    if model_name in scaled_models:
        X_used = scaler.transform(X)
    else:
        X_used = X

    # Predictions
    y_pred = model.predict(X_used)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_used)[:, 1]
    else:
        y_prob = y_pred

    # ========================================================
    # METRICS
    # ========================================================

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y, y_pred):.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")
    col5.metric("AUC", f"{roc_auc_score(y, y_prob):.4f}")
    col6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))  # smaller width, height

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False   # optional: removes side color bar to reduce width
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    
    st.subheader("Classification Report")

    report_dict = classification_report(
        y,
        y_pred,
        output_dict=True
        )

    report_df = pd.DataFrame(report_dict).transpose()

     # Round values  
    report_df = report_df.round(4)

    st.dataframe(report_df)

else:
    st.info("Upload a CSV file to begin evaluation.")
