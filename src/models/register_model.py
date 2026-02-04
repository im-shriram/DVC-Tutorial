# Standard library imports
import pathlib
import logging
from typing import Any

# Third-party imports
import pandas as pd
import joblib
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# MLflow imports
import mlflow
from mlflow import MlflowClient
import dagshub
from dotenv import load_dotenv

# Minimum threshold values for model registration - Model must exceed ALL thresholds to be promoted to production
ACCURACY_THRESHOLD = 0.5
PRECISION_THRESHOLD = 0.5
RECALL_THRESHOLD = 0.5
F1_THRESHOLD = 0.5

# Data loading
def load_data(data_dir: pathlib.Path) -> pd.DataFrame:
    """
        Load test dataset for model evaluation.
    """
    logger = logging.getLogger(name=__name__)

    # Construct path to test data
    test_path = data_dir / "features" / "test.csv"
    logger.info(f"Loading dataset from {test_path}")

    # Load and clean data (remove rows with NaN values)
    test_df = pd.read_csv(test_path).dropna()

    return test_df

# Loading latest trained model
def load_local_model(model_dir: pathlib.Path) -> Any:
    """
        Load the latest trained model from local storage.
    """
    logger = logging.getLogger(__name__)

    # Construct path to saved model
    model_path = model_dir / "models" / "bagging_classifier.joblib"
    logger.info(f"Loading local model from {model_path}")

    # Load and return the model using joblib
    return joblib.load(filename=model_path)

# Evaluate model
def evaluate_model_metrics(
    y_true: pd.Series,
    y_pred: pd.Series
) -> tuple[float, float, float, float]:
    """
        Calculate classification metrics for model evaluation.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

# Checking the metrices thresholds
def check_threshold(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float
) -> bool:
    """
        Check if model metrics exceed the minimum thresholds.
    """
    return (
        accuracy > ACCURACY_THRESHOLD
        and precision > PRECISION_THRESHOLD
        and recall > RECALL_THRESHOLD
        and f1 > F1_THRESHOLD
    )

# Registering model
def register_model(
    experiment_name: str,
    production_model_name: str,
    archive_model_name: str,
    data_dir: pathlib.Path,
    model_dir: pathlib.Path,
    client: MlflowClient,
    production_version: int,
    archive_version: int
) -> None:
    """
        Register model to MLflow Model Registry based on performance thresholds.
    """
    logger = logging.getLogger(name=__name__)

    # Load test data for evaluation    
    test_df = load_data(data_dir=data_dir)
    X = test_df.drop(columns=["sentiment", "content"])
    y = test_df["sentiment"]

    # Load the latest trained model    
    latest_model = load_local_model(model_dir=model_dir)

    # Step 3: Evaluate model performance
    y_pred = latest_model.predict(X)
    accuracy, precision, recall, f1 = evaluate_model_metrics(y, y_pred)

    logger.info(
        f"Latest model metrics - "
        f"Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1: {f1:.4f}"
    )

    # Threshold-based registration decision
    if check_threshold(accuracy, precision, recall, f1):
        # Model PASSED thresholds → Promote to production
        try:
            # Archive the current production model first
            client.set_registered_model_alias(
                name=archive_model_name,
                alias="archive",
                version=archive_version
            )

            # Remove staging alias (model was in staging before evaluation)
            client.delete_registered_model_alias(
                name=production_model_name,
                alias="staging"
            )

            # Promote new model to production
            client.set_registered_model_alias(
                name=production_model_name,
                alias="production",
                version=production_version
            )

            logger.info(
                f"Model v{production_version} promoted to production. "
                f"Previous model v{archive_version} moved to archive."
            )

        except Exception as e:
            logger.error(f"Failed to register model: {e}")

    else:
        # Model FAILED thresholds → Move directly to archive
        logger.info(
            f"Model metrics below threshold. "
            f"(Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f}). "
            f"Registration skipped - moving to archive."
        )

        # Move failed model from staging to archive
        client.set_registered_model_alias(
            name=production_model_name,
            alias="archive",
            version=production_version
        )
    
    # MLflow Model URI formats:
    """
        - Run-based:      runs:/<run_id>/<artifact_path or name>
        - Registry-based: models:/<model_name>/<version_or_alias>
    """

# Forming logger
def form_logger() -> logging.Logger:
    """
        Configure and return the root logger with console output.
    """
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    # Create console handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.INFO)

    # Define log message format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler only if not already present (prevents duplicates)
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger

def main() -> None:
    # Load environment variables from .env file
    load_dotenv()
    """
        Main entry point for the model registration pipeline.
    """
    # Initialize Logging
    logger = form_logger()
    logger.info("Starting model registration pipeline")

    # Configure MLflow Tracking    
    # Connect to DagsHub-hosted MLflow server for experiment tracking
    mlflow.set_tracking_uri(uri="https://dagshub.com/Shriram-Vibhute/Emotion-Detection-MLOps-Practices.mlflow")
    dagshub.init(repo_owner='Shriram-Vibhute', repo_name='Emotion-Detection-MLOps-Practices', mlflow=True)

    # Create MLflow client for Model Registry API calls
    client = MlflowClient()

    # Set Up Directory Paths
    # Navigate from src/models/ to project root
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_path = home_dir / "data"
    model_path = home_dir / "models"

    # Experimentation Variables
    # NOTE: Update these values for each new experiment run
    experiment_name = "emotion_detection_model_registry"
    production_model_name = "bagging_classifier"
    archive_model_name = "bagging_classifier"

    # Version numbers for alias assignment
    # production_version: Version to promote to production (if thresholds pass)
    # archive_version: Current production version to archive
    production_version = 6
    archive_version = 5

    # Execute Model Registration
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

    logger.info("Model registration pipeline completed")


if __name__ == "__main__":
    main()