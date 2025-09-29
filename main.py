import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

# -------------------------
# Dataset
# -------------------------
X, y = make_classification(n_samples=5000, n_features=20, n_informative=10,
                           n_redundant=5, n_clusters_per_class=2, weights=[0.9, 0.1],
                           flip_y=0.01, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# -------------------------
# Models to train
# -------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

mlflow.set_experiment("Fraud_Detection")

# -------------------------
# Train and log models
# -------------------------
runs_info = []

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_pred_prob))  # Convert to float
        
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name="FraudDetectionModel")
        
        runs_info.append({"run_id": run.info.run_id, "name": name, "auc": auc})
        print(f"{name} logged with AUC: {auc:.4f}")

# -------------------------
# Promote models using tags 
# -------------------------
best_run = max(runs_info, key=lambda x: x["auc"])
second_run = sorted(runs_info, key=lambda x: x["auc"], reverse=True)[1] if len(runs_info) > 1 else None

# Add tags for visualization in MLflow UI
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_tag(best_run["run_id"], "environment", "Production")
if second_run:
    client.set_tag(second_run["run_id"], "environment", "Staging")

print("\n✅ All models are logged. Open MLflow UI to monitor:")

# -------------------------
# How to open MLflow UI
# -------------------------
print("Run in terminal:")
print("mlflow ui")
print("Then go to http://127.0.0.1:5000 to see experiments, runs, models, metrics, and tags.")

