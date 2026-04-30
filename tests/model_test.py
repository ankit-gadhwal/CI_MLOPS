import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import os
import pandas as pd

# Load Dagshub token from environment variables for secure access
# The Dagshub token is required for authentication when interacting with the DAgshub Mlflow server
dagshub_token = os.getenv("CI_MLOPS")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# set the environment variables for secure access
# The dagshub token is required for authenticating with Mlflow
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name='CI_MLOPS'
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Specify the name of the model that we want to load and test
model_name = "water_potability_final1"   ## registered model name 

class TestModelLoading(unittest.TestCase):
    """unit test class to verify MLflow model loading from the Staging stage"""
    
    def test_model_in_staging(self):
        """Test if the model exists in the Staging Stage"""

        # Initialize the Mlflow client to interact with Mlflow server
        client = MlflowClient()

        # Retrive the latest versions of the models in the 'Staging' stage
        versions = client.get_latest_versions(model_name,stages=["Staging"])

        # Assert that at least one version of the model exists in the 'Staging' stage.
        # If no versions are found,it will raise an error 
        self.assertGreater(len(versions),0,"No model found in the 'Staging' stage")
    def test_model_loading(self):
        """Test if the model can be loaded properly from the Staging stage."""

        # Initialize the Mlflow client again to interact with the server
        client = MlflowClient() 
        # Retrive the latest versions of the models in the 'Staging' stage
        versions = client.get_latest_versions(model_name,stages=["Staging"])

        # If no versions are found,fails the test and skip the modal loading part
        if not versions:
            self.fail("No model found in the 'Staging' stage,skipping model loading test.")

            # get the version details of the latest model in the 'Staging' stage
            latest_version = versions[0].version
            run_id = versions[0].run_id

            # construct the String needed to load the model using its run id
            logged_model = f"runs:/{run_id}/{model_name}"

            try:
                # try to load the model from the specified path
                loaded_model = mlflow.pyfunc.load_model(logged_model)
            except Exception as e:
                # If loading the modals fails,fail the test and output the error message
                self.fail(f"Failed to load the model:{e}")

            self.assertIsNotNone(loaded_model, "The loaded model is None")
            print(f"Model successfully loaded from {logged_model}")
    
if __name__ == "__main__":
    unittest.main()



