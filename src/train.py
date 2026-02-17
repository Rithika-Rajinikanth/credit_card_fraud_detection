import kagglehub
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -------------------------------
# Load data
# -------------------------------
dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(dataset_path, "creditcard.csv")

df = pd.read_csv(csv_path)


X = df.drop("Class", axis=1)
y = df["Class"]

# Feature engineering
X["Hour"] = (X["Time"] // 3600) % 24
X = X.drop(columns=["Time"])

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------------
# Handle imbalance
# -------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

smote = SMOTE(sampling_strategy=0.02, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------------------------------
# Train model
# -------------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42
)

model.fit(X_train_res, y_train_res)

# -------------------------------
# Threshold tuning
# -------------------------------
y_prob = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

pr = pd.DataFrame({
    "threshold": thresholds,
    "precision": precision[:-1],
    "recall": recall[:-1]
})

best_threshold = pr[(pr.recall >= 0.90) & (pr.precision >= 0.05)].iloc[0].threshold

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Chosen threshold:", best_threshold)

# -------------------------------
# Save artifacts
# -------------------------------
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(best_threshold, "artifacts/threshold.pkl")

print("Artifacts saved to /artifacts")
