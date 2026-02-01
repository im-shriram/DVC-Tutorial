# Importing necessary libraries
import mlflow
from mlflow import MlflowClient
import pathlib
import logging
import pandas as pd
from typing import Any
import joblib
from sklearn.metrics import f1_score

# Loading Data
def load_data(data_dir: pathlib.Path) -> pd.DataFrame:
    logger = logging.getLogger(name=__name__)
    test_path = data_dir / "features" / "test.csv"
    logger.info(f"Loading dataset from {test_path}")
    test_df = pd.read_csv(test_path).dropna()
    return test_df

# Loading Local Model
def load_local_model(model_dir: pathlib.Path) -> Any:
    logger = logging.getLogger(__name__)
    model_path = model_dir / "models" / "bagging_classifier.joblib"
    logger.info(f"Loading local model from {model_path}")
    return joblib.load(filename=model_path)

def get_latest_run_info(experiment_name: str) -> tuple[str, str]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    print(runs.columns)
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{run_id}/bagging_classifier"
    return model_uri, run_id

def register_model(experiment_name: str, production_model_name: str, archive_model_name: str, data_dir: pathlib.Path, model_dir: pathlib.Path, client: MlflowClient, production_version: int, archive_version: int) -> None:
    logger = logging.getLogger(name=__name__)

    # Loading test data
    test_df = load_data(data_dir = data_dir)
    X = test_df.drop(columns=["sentiment","content"])
    y = test_df["sentiment"]

    # 1. Load the latest local model (just trained)
    latest_model = load_local_model(model_dir=model_dir)
    latest_model_uri, latest_run_id = get_latest_run_info(experiment_name)
    
    # 2. Try to load the PRODUCTION model
    production_model = None
    try:
        # Note: Using aliases or stages like "Production" is more reliable
        production_model_uri = f"models:/{production_model_name}@production" 
        production_model = mlflow.pyfunc.load_model(model_uri=production_model_uri)
        logger.info("Successfully loaded production model for comparison.")
    except Exception as e:
        logger.warning(f"No production model found: {e}")

    # 3. Decision Logic
    should_register = False
    if production_model is None:
        logger.info("No production model exists. Registering as first version.")
        should_register = True
    else:
        logger.info("Calculating performance comparison...")
        latest_f1 = f1_score(y, latest_model.predict(X))
        prod_f1 = f1_score(y, production_model.predict(X))
        
        logger.info(f"Latest F1: {latest_f1}, Production F1: {prod_f1}")
        if latest_f1 >= prod_f1:
            logger.info("Latest model is better or equal. Registering...")
            client.set_registered_model_alias(name=archive_model_name, alias="archive", version=archive_version)

            logger.info("Shifting production model to archive")
            should_register = True
        else:
            logger.info("Production model is better. Skipping registration.")

    # 4. Perform Registration
    if should_register:
        try:
            mlflow.register_model(model_uri=latest_model_uri, name=production_model_name)
            client.set_registered_model_alias(name=production_model_name, alias="production", version=production_version) # NOTE: The production alias automatically transferred from v1 to v6
            logger.info("Model registered successfully.")
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            logger.error("TIP: Ensure evaluate_model.py uses mlflow.sklearn.log_model(model, artifact_path='model')")

# Forming Logger
def form_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(console_handler)
    return logger

def main() -> None:
    logger = form_logger()
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    client = MlflowClient()

    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_path = home_dir / "data"
    model_path = home_dir / "models"

    # Need to change for every run
    experiment_name = "sentement_analysis_experiment_tracking"
    production_model_name = "bagging_classifier"
    archive_model_name = "bagging_classifier"
    production_version = 4
    archive_version = 3 

    register_model(
        experiment_name=experiment_name,
        production_model_name=production_model_name,
        archive_model_name=archive_model_name,
        data_dir=data_path,
        model_dir=model_path,
        client=client,
        production_version=production_version,
        archive_version=archive_version
    )

if __name__ == "__main__":
    main()