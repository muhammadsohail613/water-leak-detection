import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# Load data
df = pd.read_csv("data/sensor_data.csv")

features = ["flow_rate_lpm", "pressure_bar", "velocity_ms"]
X = df[features].values
y = df["label"].values  # 0=normal, 1=leak

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest (unsupervised — realistic for real deployments)
model = IsolationForest(
    n_estimators=200,
    contamination=0.10,   # ~10% anomaly rate matches our injected data
    random_state=42
)
model.fit(X_scaled)

# Predict: IsolationForest returns -1 (anomaly) or 1 (normal)
raw_preds = model.predict(X_scaled)
preds = np.where(raw_preds == -1, 1, 0)  # convert to 1=leak, 0=normal

# Anomaly scores (lower = more anomalous)
scores = model.decision_function(X_scaled)
df["anomaly_score"] = np.round(scores, 4)
df["predicted_leak"] = preds

# Severity: based on score percentile
def get_severity(score):
    if score < np.percentile(scores, 5):
        return "CRITICAL"
    elif score < np.percentile(scores, 10):
        return "HIGH"
    elif score < np.percentile(scores, 20):
        return "MEDIUM"
    else:
        return "NORMAL"

df["severity"] = df["anomaly_score"].apply(get_severity)

# Save enriched data
df.to_csv("data/predictions.csv", index=False)

# Save model + scaler
os.makedirs("model", exist_ok=True)
with open("model/isolation_forest.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Evaluation
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"\nTotal samples : {len(y)}")
print(f"Actual leaks  : {y.sum()}")
print(f"Detected leaks: {preds.sum()}")
print("\nClassification Report:")
print(classification_report(y, preds, target_names=["Normal", "Leak"]))
print("Confusion Matrix:")
cm = confusion_matrix(y, preds)
print(f"  True Normal  (TN): {cm[0][0]}")
print(f"  False Alarm  (FP): {cm[0][1]}")
print(f"  Missed Leaks (FN): {cm[1][0]}")
print(f"  Caught Leaks (TP): {cm[1][1]}")
print("\n✅ Model saved to model/isolation_forest.pkl")
print("✅ Predictions saved to data/predictions.csv")

# Preview leak detections
leaks_found = df[df["predicted_leak"] == 1][["timestamp", "flow_rate_lpm", "pressure_bar", "severity"]].head(10)
print("\nSample detected leaks:")
print(leaks_found.to_string(index=False))