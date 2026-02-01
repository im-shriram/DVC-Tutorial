# Importing necessary libraries
import numpy as np
import pandas as pd
import pathlib
import logging
import json
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve


# Loading Data - Train and Test
def load_data(data_dir: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)

    # Forming file paths
    train_path = data_dir / "features" / "train.csv"
    test_path = data_dir / "features" / "test.csv"
    logger.info(f"Loading dataset from {data_dir / "features"}")

    # Loading datasets
    train_df = pd.read_csv(filepath_or_buffer=train_path)
    test_df = pd.read_csv(filepath_or_buffer=test_path)
    
    # Logging both the datasets using mlflow
    logger.info(f"Logging Train and Test Datasets using MLflow")
    # Convert pandas DataFrames to mlflow compatible datasets
    train_dataset = mlflow.data.from_pandas(train_df.sample(5))
    test_dataset = mlflow.data.from_pandas(test_df.sample(5))
    
    # Log the datasets using mlflow
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")

    # Returning datasets
    return train_df.dropna(), test_df.dropna()

# Loading Parameters
def load_params(params_path: pathlib.Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading parameters from {params_path}")
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Check if 'make_dataset' key exists in params
        if 'model_training' not in params:
            logger.error("model_training section not found in parameters file")
            raise KeyError("model_training section not found in parameters file")

        # Logging all the parameters using mlflow
        logger.info(f"Logging all the prameters using MLflow")
        mlflow.log_params(params["data_ingestion"])
        mlflow.log_params(params["feature_engineering"])
        mlflow.log_params(params["model_training"]["estimator"])
        mlflow.log_params(params["model_training"]["bagging"])

# Loading Model
def load_model(model_dir: str, sample_input: pd.DataFrame) -> BaggingClassifier:
    logger = logging.getLogger(__name__)

    # Loading Model
    model_path = model_dir / "models" / "bagging_classifier.joblib"
    logger.info(f"Loading model from {model_path} and logging as well as registering the model")
    model = joblib.load(filename=model_path)

    # Logging model using MLflow
    logger.info(f"Logging model using MLflow from {model_path}")
    sample_output = model.predict(sample_input)
    signature = mlflow.models.infer_signature(sample_input, sample_output) # This is automatic signature. You can also define it manually by defining the datatype of every single feature in your dataframe.
    mlflow.sklearn.log_model(model, signature=signature, name="bagging_classifier")
    # NOTE: If the name of the registered model is same as previous one, then it will be considered as new version of that same model.

    client = mlflow.tracking.MlflowClient()
    registered_model_name = "bagging_classifier"
    model_version = client.get_latest_versions(registered_model_name, stages=["None"])[0]
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="staging",  # You can use: "champion", "staging", "production", etc.
        version=model_version.version
    )

    # Set tag and description on the latest model version
    client.set_model_version_tag(
        name=registered_model_name,
        version=model_version.version,
        key="created_by",
        value="Shriram"
    )

    # Updating the model version
    """
        client.update_model_version(
            name=registered_model_name,
            version=latest_version.version,
            description="Latest bagging classifier model with 50 BOW features evaluated on train/test splits"
        )
    """

    # Why model signatures are needed?
    return model

# Evaluating Model
def evaluate_model(df: pd.DataFrame, model: BaggingClassifier, split: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Evaluating the model")

    X = df.drop(columns=["sentiment", "content"])
    y = df["sentiment"]

    y_pred = model.predict(X=X)
    y_probab = model.predict_proba(X=X)
    predictions = y_probab[:, 1]

    report_dict = classification_report(y_true=y, y_pred=y_pred, output_dict=True)
    clf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)

    # Log metrics for each class
    for class_label in report_dict.keys():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics_prefix = f"{split}_class_{class_label}"
            mlflow.log_metric(key=f"{metrics_prefix}_precision", value=report_dict[class_label]['precision'])
            mlflow.log_metric(key=f"{metrics_prefix}_recall", value=report_dict[class_label]['recall'])
            mlflow.log_metric(key=f"{metrics_prefix}_f1", value=report_dict[class_label]['f1-score'])
            mlflow.log_metric(key=f"{metrics_prefix}_support", value=report_dict[class_label]['support'])
    
    # Log overall metrics
    mlflow.log_metric(f"{split}_accuracy", report_dict['accuracy'])
    
    # Log macro averages
    mlflow.log_metric(f"{split}_macro_precision", report_dict['macro avg']['precision'])
    mlflow.log_metric(f"{split}_macro_recall", report_dict['macro avg']['recall'])
    mlflow.log_metric(f"{split}_macro_f1", report_dict['macro avg']['f1-score'])
    
    # Log weighted averages
    mlflow.log_metric(f"{split}_weighted_precision", report_dict['weighted avg']['precision'])
    mlflow.log_metric(f"{split}_weighted_recall", report_dict['weighted avg']['recall'])
    mlflow.log_metric(f"{split}_weighted_f1", report_dict['weighted avg']['f1-score'])

    # Log confusion matrix plot using MLflow
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=clf_matrix, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix - {split} set')
    mlflow.log_figure(fig, f"{split}_confusion_matrix.png")
    plt.close()

    # Log precision-recall curve plot using MLflow
    precision, recall, _ = precision_recall_curve(y_true=y, y_score=predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax)
    plt.title(f'Precision-Recall Curve - {split} set')
    mlflow.log_figure(fig, f"{split}_precision_recall_curve.png")
    plt.close()

# Forming Logger
def form_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# Main function
def main() -> None:
    # Forming logger
    logger = form_logger()
    logger.info(msg="Started model evaluation pipeline")

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'http://127.0.0.1:8080')

    # Forming directory paths
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    params_path = home_dir / "params.yaml"
    model_dir = home_dir / "models" 
    logger.info(f"Working directory: {home_dir}")

    # dvclive storing path
    dvclive_path = home_dir / "dvclive"

    # Creating new experiment
    if mlflow.get_experiment_by_name(name="sentement_analysis_experiment_tracking") is None:
        experiment_id = mlflow.create_experiment(name="sentement_analysis_experiment_tracking") # If experiment already exists then dont create it again otherwise throw error. # TODO: Handle exception if experiment already exists.
    else: 
        experiment_id = mlflow.get_experiment_by_name(name="sentement_analysis_experiment_tracking").experiment_id

    tags = {
        "engineering": "ML Platform",
        "release.candidate": "Shriram",
        "release.version": "1.0.0",
        "model.type": "BaggingClassifier",
        "feature.engineering": "Bag of Words"
    }   
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"bagging_classifier_500_bow_features_run_3", # TODO: dynamic featching no. features from params.yaml
        tags=tags,
        nested=False,
        description="Model evaluation run for bagging classifier with 500 bag of words features") as run:
        # Logging tags
        mlflow.set_tags(tags)
        
        # Loading data
        train_df, test_df = load_data(data_dir=data_dir)

        # Loading Parameters
        load_params(params_path=params_path)

        # Loading model
        model = load_model(model_dir=model_dir, sample_input=train_df.dropna().drop(columns=["sentiment", "content"]).iloc[:3, :])

        # Evaluating model
        evaluate_model(df=train_df, model=model, split="train")
        evaluate_model(df=test_df, model=model, split="test")

        logger.info("Model Evaluation completed successfully")

if __name__ == "__main__":
    main()