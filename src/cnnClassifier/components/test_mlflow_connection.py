import os
import sys
import mlflow

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def test_connection():
    """
    Test connectivity to MLflow remote tracking server.
    """
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        print("❌ MLFLOW_TRACKING_URI is not set.")
        print("Please set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD in your environment or .env file.")
        return False

    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_name="connection_test"):
            mlflow.log_param("test", "hello")
            mlflow.log_metric("dummy_metric", 1.0)
        print("✅ Successfully logged to DagsHub MLflow!")
        return True
    except Exception as e:
        print(f"❌ MLflow connection failed: {e}")
        return False


if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)
