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
# The dagshub token is required for authentication when interacting with the 