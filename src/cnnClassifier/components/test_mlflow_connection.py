import os
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/Pratham415/Kidney-Disease-classification.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'fbea9abf7cb94ad9ddad624ef8b47700d5f8f126'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'fbea9abf7cb94ad9ddad624ef8b47700d5f8f126'

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

with mlflow.start_run(run_name="connection_test"):
    mlflow.log_param("test", "hello")
    mlflow.log_metric("dummy_metric", 1.0)

print("✅ Successfully logged to DagsHub MLflow!")