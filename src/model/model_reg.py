import mlflow
import dagshub
import json
from mlflow.tracking import MlflowClient

# 1. Setup Authentication
# dagshub.init(repo_owner='ankit-gadhwal', repo_name='CI_MLOPS', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/ankit-gadhwal/CI_MLOPS.mlflow")
# client = MlflowClient()

import os
# load Dagshub token from environment variables
dagshub_token = os.getenv("CI_MLops")
if not dagshub_token:
    raise EnvironmentError("DAGSEUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# DagsHub repository details
dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name='CI_MLOPS'
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final_model_reg")
# 2. Load Run Info
with open("reports/run_info.json", 'r') as f:
    run_info = json.load(f)

run_id = run_info['run_id']
# ⚠️ IMPORTANT: In your printout, it was 'model_artifact' 
# Ensure this matches the folder name in DagsHub UI exactly
artifact_path = run_info["model_name"]
registry_name = "water_potability_final1"

# 3. Construct URI
model_uri = f"runs:/{run_id}/{artifact_path}"
print(f"Registering model from: {model_uri}")

try:
    # 4. Register the model using the Client (more stable for remote)
    # This creates the version directly
    result = client.create_model_version(
        name=registry_name,
        source=model_uri,
        run_id=run_id
    )
    
    version = result.version
    print(f"✅ Registered Version: {version}")

    # 5. Transition to Staging
    client.transition_model_version_stage(
        name=registry_name,
        version=str(version),
        stage="production",
        archive_existing_versions=True
    )
    print(f"🚀 Model {registry_name} v{version} is now in STAGING.")

except Exception as e:
    print(f"❌ Registration failed: {e}")
    print("If you see '404', please verify that the folder 'model_artifact' contains an 'MLmodel' file on DagsHub.")
