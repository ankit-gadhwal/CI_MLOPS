import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import os
import pandas as pd

# Load Dagshub token from environment variables for secure access
# The Dagshub token is required for authentication when interacting with the DAgshub Mlflow server
dagshub_token = os.getenv("CI_MlOPS")
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
