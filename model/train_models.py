"""
train_models.py

STEP 2 - Train 6 ML Models and Compute Metrics

This script performs the following:

1. Loads Kaggle Breast Cancer Wisconsin dataset from /data/data.csv
2. Removes unnecessary columns
3. Encodes diagnosis column
4. Splits data into train/test sets
5. Scales numerical features
6. Trains 6 classification models:
      - Logistic Regression
      - Decision Tree
      - KNN
      - Naive Bayes
      - Random Forest
      - XGBoost
7. Evaluates models using:
      - Accuracy
      - AUC
      - Precision
      - Recall
      - F1 Score
      - Matthews Correlation Coefficient (MCC)
8. Saves:
      - trained models (*.pkl)
      - scaler.pkl
      - metrics.csv

Run from project root:

    python model/train_models.py
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from xgboost import XGBClassifier

import joblib


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "data.csv")


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==========================================")
print("LOADING DATASET")
print("==========================================")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"data.csv not found in {DATA_DIR}. "
        "Make sure Kaggle file is saved as /data/data.csv"
    )

df = pd.read_csv(DATA_FILE)

print("Dataset loaded.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# DATA CLEANING
# ============================================================

print("\n==========================================")
print("DATA CLEANING")
print("==========================================")

# Drop ID column if present
df.drop(columns=["id"], inplace=True, errors="ignore")

# Drop unnamed column common in Kaggle version
df.drop(columns=["Unnamed: 32"], inplace=True, errors="ignore")

# Encode diagnosis column
if "diagnosis" in df.columns:
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
else:
    raise ValueError("Target column 'diagnosis' not found!")

# Missing values check
print("\nMissing values per column:")
print(df.isnull().sum())

if df.isnull().sum().sum() > 0:
    print("\n⚠️ Missing values detected. Dropping rows.")
    df.dropna(inplace=True)

print("\nCleaned dataset shape:", df.shape)


# ============================================================
# FEATURE / TARGET SPLIT
# ============================================================

print("\n==========================================")
print("FEATURE / TARGET SPLIT")
print("==========================================")

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

print("Feature matrix shape:", X.shape)
print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n==========================================")
print("TRAIN / TEST SPLIT")
print("==========================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# FEATURE SCALING
# ============================================================

print("\n==========================================")
print("FEATURE SCALING")
print("==========================================")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

print("\n==========================================")
print("INITIALIZING MODELS")
print("==========================================")

lr = LogisticRegression(max_iter=1000)

dt = DecisionTreeClassifier(
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=7
)

nb = GaussianNB()

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODELS
# ============================================================

print("\n==========================================")
print("TRAINING MODELS")
print("==========================================")

print("Training Logistic Regression...")
lr.fit(X_train_scaled, y_train)

print("Training Decision Tree...")
dt.fit(X_train, y_train)

print("Training KNN...")
knn.fit(X_train_scaled, y_train)

print("Training Naive Bayes...")
nb.fit(X_train_scaled, y_train)

print("Training Random Forest...")
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb.fit(X_train, y_train)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

print("\n==========================================")
print("EVALUATING MODELS")
print("==========================================")

def evaluate_model(model, X_test, y_test):
    """
    Returns dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Probability predictions for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


# ============================================================
# COMPUTE METRICS FOR ALL MODELS
# ============================================================

results = {}

results["Logistic Regression"] = evaluate_model(
    lr, X_test_scaled, y_test
)

results["Decision Tree"] = evaluate_model(
    dt, X_test, y_test
)

results["KNN"] = evaluate_model(
    knn, X_test_scaled, y_test
)

results["Naive Bayes"] = evaluate_model(
    nb, X_test_scaled, y_test
)

results["Random Forest"] = evaluate_model(
    rf, X_test, y_test
)

results["XGBoost"] = evaluate_model(
    xgb, X_test, y_test
)


# ============================================================
# SAVE METRICS TABLE
# ============================================================

results_df = pd.DataFrame(results).T.round(4)

print("\n==========================================")
print("FINAL METRICS TABLE")
print("==========================================")
print(results_df)

metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
results_df.to_csv(metrics_path)

print(f"\nMetrics saved to: {metrics_path}")


# ============================================================
# SAVE MODELS + SCALER
# ============================================================

print("\n==========================================")
print("SAVING MODELS")
print("==========================================")

joblib.dump(lr, os.path.join(MODEL_DIR, "logistic.pkl"))
joblib.dump(dt, os.path.join(MODEL_DIR, "decision_tree.pkl"))
joblib.dump(knn, os.path.join(MODEL_DIR, "knn.pkl"))
joblib.dump(nb, os.path.join(MODEL_DIR, "naive_bayes.pkl"))
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost.pkl"))

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("All models saved in /model folder.")
print("Scaler saved.")
