import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

df = pd.read_csv("data/transactions.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

model = joblib.load("artifacts/model.pkl")

y_prob = model.predict_proba(X)[:, 1]

precision, recall, thresholds = precision_recall_curve(y, y_prob)

pr = pd.DataFrame({
    "threshold": thresholds,
    "precision": precision[:-1],
    "recall": recall[:-1]
})

best_threshold = pr[pr.recall >= 0.90].iloc[0].threshold
joblib.dump(best_threshold, "artifacts/threshold.pkl")

y_pred = (y_prob >= best_threshold).astype(int)

print("Threshold:", best_threshold)
print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))
