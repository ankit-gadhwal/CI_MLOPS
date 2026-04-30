import mlflow
import dagshub
import json
from mlflow.tracking import MlflowClient
import os
# 1. Setup
# dagshub.init(repo_owner='ankit-gadhwal', repo_name='CI_MLOPS', mlflow=True)
dagshub_token = os.getenv("CI_MLOPS")
if not dagshub_token:
    raise EnvironmentError("DAGSEUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# # DagsHub repository details
dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name='CI_MLOPS'
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final Model")
client = MlflowClient()

# 2. Load the Run ID you just created
with open("reports/run_info.json", 'r') as f:
    run_info = json.load(f)

run_id = run_info['run_id']
# This MUST match the folder name you saw in the UI
model_path = run_info["model_name"] 
registry_name = "water_potability_final1"

# 3. Construct the URI
model_uri = f"runs:/{run_id}/{model_path}"
print(f"Registering model from: {model_uri}")

try:
    # 4. Register the model version
    # Since the MLmodel file exists, this will now succeed
    result = client.create_model_version(
        name=registry_name,
        source=model_uri,
        run_id=run_id
    )
    
    # 5. Move to Staging for CI/CD testing
    client.transition_model_version_stage(
        name=registry_name,
        version=result.version,
        stage="Staging",
        archive_existing_versions=True
    )
    
    print(f"✅ SUCCESS! Model version {result.version} is now in STAGING.")

except Exception as e:
    print(f"❌ Registration failed: {e}")
