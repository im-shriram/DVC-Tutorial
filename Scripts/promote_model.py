import mlflow
import dagshub
import os

def promote_model():
    # Set up MLflow tracking URI
    """
        mlflow.set_tracking_uri(uri="https://dagshub.com/Shriram-Vibhute/Emotion-Detection-MLOps-Practices.mlflow")
        dagshub.init(repo_owner='Shriram-Vibhute', repo_name='Emotion-Detection-MLOps-Practices', mlflow=True)
    """

    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Shriram-Vibhute"
    repo_name = "Emotion-Detection-MLOps-Practices"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    # Get the latest version in staging
    client = mlflow.MlflowClient()
    model_name = "bagging_classifier"
    
    try: 
        if len(client.get_model_version_by_alias(model_name, "production")) >= 1:
            previous_version_staging = client.get_model_version_by_alias(model_name, "production")[0].version
            client.delete_registered_model_alias(
                name=model_name,
                alias="production"
            )
            client.set_registered_model_alias(
                name=model_name,
                alias="archived",
                version=previous_version_staging
            )
        else:
            previous_version_staging = None
    
    except Exception as e:
        print(f"There are no any models in Production stage.")

    latest_version_staging = client.get_latest_versions(model_name, stages=["None"])[0].version

    try:
        if "staging" in client.get_latest_versions(model_name, stages=["None"])[0].aliases:
            client.delete_registered_model_alias(
                name=model_name,
                alias="staging"
            )
    except Exception as e:
        print(f"Model version {latest_version_staging} does not have staging alias.")

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version_staging
    )

    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()