import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd

# 1. Setup Authentication
dagshub_token = os.getenv("CI_MLOPS")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable (CI_MLOPS) is not set")

# Set environment variables for MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name = "CI_MLOPS"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Registered model name in DagsHub
model_name = "water_potability_final1" 

class TestModelLoading(unittest.TestCase):
    """Unit test class to verify MLflow model loading and performance from Staging."""

    def test_model_in_staging(self):
        """Verify that at least one version of the model exists in Staging."""
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        self.assertGreater(len(versions), 0, f"No model versions found in 'Staging' for {model_name}")

    def test_model_loading(self):
        """Test if the model can be successfully loaded from the Staging stage."""
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not versions:
            self.fail("No model found in Staging, skipping loading test.")

        # ✅ FIX: Use the specific Run ID and the correct artifact folder name
        run_id = versions[0].run_id
        logged_model = f"runs:/{run_id}/best_Model"

        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
            self.assertIsNotNone(loaded_model, "The loaded model is None")
            print(f"✅ Model successfully loaded from {logged_model}")
        except Exception as e:
            self.fail(f"Failed to load the model from {logged_model}: {e}")

    # def test_model_performance(self):
    #     """Test model performance against predefined thresholds using test data."""
    #     client = MlflowClient()
    #     versions = client.get_latest_versions(model_name, stages=["Staging"])

    #     if not versions:
    #         self.fail("No model found in Staging, skipping performance test.")

    #     # ✅ FIX: Get Run ID and use 'best_Model' folder
    #     run_id = versions[0].run_id
    #     logged_model = f"runs:/{run_id}/best_Model"
        
    #     try:
    #         loaded_model = mlflow.pyfunc.load_model(logged_model)
    #     except Exception as e:
    #         self.fail(f"Could not load model for performance test: {e}")

    #     # Load test data
    #     test_data_path = "./data/processed/test_processed.csv"
    #     if not os.path.exists(test_data_path):
    #         self.fail(f"Test data not found at {test_data_path}")

    #     test_data = pd.read_csv(test_data_path)
    #     X_test = test_data.drop(columns=["potability"])
    #     y_test = test_data["potability"]

    #     # Make predictions and calculate metrics
    #     predictions = loaded_model.predict(X_test)

    #     accuracy = accuracy_score(y_test, predictions)
    #     precision = precision_score(y_test, predictions, average="binary")
    #     recall = recall_score(y_test, predictions, average="binary")
    #     f1 = f1_score(y_test, predictions, average="binary")

    #     print(f"--- Performance Results ---")
    #     print(f"Accuracy:  {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall:    {recall:.4f}")
    #     print(f"F1 Score:  {f1:.4f}")

    #     # Threshold assertions
    #     self.assertGreaterEqual(accuracy, 0.7, f"Accuracy {accuracy} is below 0.7")
    #     self.assertGreaterEqual(precision, 0.3, f"Precision {precision} is below 0.3")
    #     self.assertGreaterEqual(recall, 0.3, f"Recall {recall} is below 0.3")
    #     self.assertGreaterEqual(f1, 0.3, f"F1 Score {f1} is below 0.3")

if __name__ == "__main__":
    unittest.main()
