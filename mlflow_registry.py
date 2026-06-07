import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from mlflow.tracking import MlflowClient

# Local tracking server configurations
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Production_Lifecycle")

# Baseline ML model step (Automated Workflow components)
iris = load_iris()
model = RandomForestClassifier(n_estimators=100)
model.fit(iris.data, iris.target)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    # Log model artifacts into MLflow Model Registry
    model_info = mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="iris-classifier",
        registered_model_name="IrisPredictorModel"
    )
    print(f"Model logged under Run ID: {run_id}")

# Connect with Central Model Registry Client
client = MlflowClient()
model_name = "IrisPredictorModel"

# Fetch the latest version inside registry layers
latest_versions = client.get_latest_versions(model_name, stages=["None"])
latest_version = latest_versions[0].version
print(f"Registered Model Version inside Registry: {latest_version}")

# STEP 5.2 Transition Stage Automation: Staging -> Production
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)
print(f"SUCCESS: Model Version {latest_version} transitioned directly to PRODUCTION stage!")