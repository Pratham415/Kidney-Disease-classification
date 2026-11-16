import os
import sys
# pyrefly: ignore[missing-import]
import mlflow

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
if not tracking_uri:
    print("❌ MLFLOW_TRACKING_URI is not set.")
    print("Please set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD in your environment or .env file.")
    sys.exit(1)

mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run(run_name="connection_test"):
    mlflow.log_param("test", "hello")
    mlflow.log_metric("dummy_metric", 1.0)

print("✅ Successfully logged to DagsHub MLflow!")