# Step 1: Import all required libraries for data handling, modeling, and tracking
import pandas as pd
import pathlib
import logging
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import json
import mlflow
import dagshub
from mlflow import MlflowClient
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve


# Function to load training and testing datasets from the features directory
def load_data(data_dir: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)

    # Create full file paths for the train and test CSV files
    train_path = data_dir / "features" / "train.csv"
    test_path = data_dir / "features" / "test.csv"
    logger.info(f"Searching for data files in the directory: {data_dir / 'features'}")

    # Read the data from CSV files into pandas DataFrames
    logger.info("Reading train.csv and test.csv files...")
    train_df = pd.read_csv(filepath_or_buffer=train_path)
    test_df = pd.read_csv(filepath_or_buffer=test_path)
    
    # Log small samples of the data to MLflow for experiment tracking visibility
    logger.info("Converting a sample of the data for MLflow logging purposes")
    # We sample 5 rows from each dataset to keep the log lightweight but informative
    train_dataset = mlflow.data.from_pandas(train_df.sample(5))
    test_dataset = mlflow.data.from_pandas(test_df.sample(5))
    
    # Send the sample dataset metadata to MLflow tracking server
    logger.info("Logging dataset samples to MLflow tracking server")
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")

    # Drop any missing values to ensure the model receives clean data, then return them
    return train_df.dropna(), test_df.dropna()


# Function to load hyperparameters from the configuration file
def load_params(params_path: pathlib.Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Opening and reading configuration from: {params_path}")
    
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Verify that the necessary configuration section exists
        if 'model_training' not in params:
            logger.error("The critical 'model_training' section is missing from the params.yaml file!")
            raise KeyError("model_training section not found in parameters file")

        # Log all configuration parameters to MLflow so we know exactly what settings were used for this run
        logger.info("Successfully loaded parameters. Now logging them to MLflow for run documentation")
        mlflow.log_params(params["data_ingestion"])
        mlflow.log_params(params["feature_engineering"])
        mlflow.log_params(params["model_training"]["estimator"])
        mlflow.log_params(params["model_training"]["bagging"])


# Function to load the saved model and register it in the MLflow model registry
def load_model(model_dir: str, sample_input: pd.DataFrame, client: MlflowClient) -> BaggingClassifier:
    logger = logging.getLogger(__name__)

    # Define the path where the trained model is stored
    model_path = model_dir / "models" / "bagging_classifier.joblib"
    logger.info(f"Accessing the saved model file at: {model_path}")
    
    # Load the model object from the disk
    model = joblib.load(filename=model_path)

    # Start the model registration process with MLflow
    logger.info("Initializing model registration and signature creation in MLflow")
    
    # Use a small set of data to let MLflow automatically understand what input format the model expects
    sample_output = model.predict(sample_input)
    signature = mlflow.models.infer_signature(sample_input, sample_output) 
    
    # Save the model object and its signature to the MLflow server and register it with a name
    mlflow.sklearn.log_model(model, signature=signature, name="bagging_classifier", registered_model_name="bagging_classifier") 

    # Assign a 'staging' alias to this model version so we know it is ready for further testing
    logger.info("Assigning 'staging' alias to the newly registered model in the registry")
    client.set_registered_model_alias(name="bagging_classifier", alias="staging", version=2)

    # Note on Model Signatures:
    """
        A model signature is like a contract. It ensures that any application trying to use this model 
        provides data in correctly named columns and expected data types. This prevents errors when 
        integrating the model with other systems or APIs.
    """

    return model


# Function to evaluate performance on a given dataset and log results
def evaluate_model(df: pd.DataFrame, model: BaggingClassifier, split: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Beginning performance evaluation on the '{split}' data set")

    # Split the data into features (X) and actual labels (y)
    X = df.drop(columns=["sentiment", "content"])
    y = df["sentiment"]

    # Use the model to predict labels and calculate probability scores
    logger.info(f"Generating predictions and probability scores for the '{split}' set")
    y_pred = model.predict(X=X)
    y_probab = model.predict_proba(X=X)
    predictions = y_probab[:, 1] # Extracting scores for the positive class

    # Generate a detailed classification report and a confusion matrix
    report_dict = classification_report(y_true=y, y_pred=y_pred, output_dict=True)
    clf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)

    # Individually log precision, recall, and f1-score for every single class label
    logger.info(f"Logging detailed per-class metrics to MLflow for '{split}' set")
    for class_label in report_dict.keys():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics_prefix = f"{split}_class_{class_label}"
            mlflow.log_metric(key=f"{metrics_prefix}_precision", value=report_dict[class_label]['precision'])
            mlflow.log_metric(key=f"{metrics_prefix}_recall", value=report_dict[class_label]['recall'])
            mlflow.log_metric(key=f"{metrics_prefix}_f1", value=report_dict[class_label]['f1-score'])
            mlflow.log_metric(key=f"{metrics_prefix}_support", value=report_dict[class_label]['support'])
    
    # Log the general performance metrics like accuracy and averages
    mlflow.log_metric(f"{split}_accuracy", report_dict['accuracy'])
    
    # Macro avg: calculates metrics independently for each class then takes the average
    mlflow.log_metric(f"{split}_macro_precision", report_dict['macro avg']['precision'])
    mlflow.log_metric(f"{split}_macro_recall", report_dict['macro avg']['recall'])
    mlflow.log_metric(f"{split}_macro_f1", report_dict['macro avg']['f1-score'])
    
    # Weighted avg: calculates metrics for each class, but accounts for how many samples each class has
    mlflow.log_metric(f"{split}_weighted_precision", report_dict['weighted avg']['precision'])
    mlflow.log_metric(f"{split}_weighted_recall", report_dict['weighted avg']['recall'])
    mlflow.log_metric(f"{split}_weighted_f1", report_dict['weighted avg']['f1-score'])

    # Visualize results with a Confusion Matrix Plot
    logger.info(f"Generating and logging the Confusion Matrix visualization for '{split}' set")
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=clf_matrix, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix - {split} set')
    mlflow.log_figure(fig, f"{split}_confusion_matrix.png")
    plt.close()

    # Visualize results with a Precision-Recall Curve Plot
    logger.info(f"Generating and logging the Precision-Recall Curve visualization for '{split}' set")
    precision, recall, _ = precision_recall_curve(y_true=y, y_score=predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax)
    plt.title(f'Precision-Recall Curve - {split} set')
    mlflow.log_figure(fig, f"{split}_precision_recall_curve.png")
    plt.close()


# Function to save model identifier and path to a local JSON file for record-keeping
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Saving run ID and model path for future reference in: {file_path}")
    try:
        # Create a dictionary with the info and write it to a JSON file
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f'Successfully wrote model information to {file_path}')
    except Exception as e:
        logger.error(f'Failed to save model information! Technical details: {e}')
        raise


# Function to configure the system logger for terminal output
def form_logger() -> logging.Logger:
    logger = logging.getLogger()
    # Set the threshold for logging set to DEBUG to capture all details
    logger.setLevel(level=logging.DEBUG)

    # Attach a console handler to print messages to the computer terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # Define a clear format: [Time] - [Logger Name] - [Level] - [Message]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Only add the handler if one doesn't exist to avoid printing the same message multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# Main entry point for the evaluation process
def main() -> None:
    # Initialize the logger to track progress
    logger = form_logger()
    logger.info("The model evaluation pipeline has officially started")

    # Connect to DagsHub for remote MLflow tracking
    logger.info("Initializing connection to DagsHub and setting up MLflow tracking URI")
    mlflow.set_tracking_uri(uri="https://dagshub.com/Shriram-Vibhute/Emotion-Detection-MLOps-Practices.mlflow")
    dagshub.init(repo_owner='Shriram-Vibhute', repo_name='Emotion-Detection-MLOps-Practices', mlflow=True)
    client = MlflowClient()

    # Determine project root and directory paths for data, params, and models
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    params_path = home_dir / "params.yaml"
    model_dir = home_dir / "models" 
    logger.info(f"Project base directory is identified as: {home_dir}")

    # Check for the existence of the experiment or create a new one
    logger.info("Setting up the MLflow experiment for emotion detection model registry")
    if mlflow.get_experiment_by_name(name="emotion_detection_model_registry") is None:
        experiment_id = mlflow.create_experiment(name="emotion_detection_model_registry") 
    experiment_id = mlflow.get_experiment_by_name(name="emotion_detection_model_registry").experiment_id

    # Metadata tags to categorize this specific experiment run
    tags = {
        "engineering": "ML Platform",
        "release.candidate": "Shriram",
        "release.version": "1.0.0",
        "model.type": "BaggingClassifier",
        "feature.engineering": "Bag of Words"
    }   
    
    # Start the actual tracking process within a 'with' block to ensure clean closure
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="emotion_detection_model_registry_run_2",
        tags=tags,
        nested=False,
        description="Model evaluation run for bagging classifier with 100 bag of words features") as run:
        
        # Attach the descriptive tags to the run
        logger.info("MLflow run started. Attaching metadata tags to the current run")
        mlflow.set_tags(tags)
        
        # Load the feature-engineered data
        logger.info("Step 1: Loading train and test data for evaluation")
        train_df, test_df = load_data(data_dir=data_dir)

        # Load the project parameters
        logger.info("Step 2: Loading and logging project configuration parameters")
        load_params(params_path=params_path)

        # Load the trained model and register it
        logger.info("Step 3: Loading the trained model and registering it in the model registry")
        model = load_model(model_dir=model_dir, sample_input=train_df.dropna().drop(columns=["sentiment", "content"]).iloc[:3, :], client=client)

        # Store run details locally in a report file
        logger.info("Step 4: Writing run details to the experiment_info report")
        save_model_info(run.info.run_id, "bagging_classifier", 'reports/experiment_info.json')

        # Perform the actual evaluation on both datasets
        logger.info("Step 5: Beginning model performance assessment")
        evaluate_model(df=train_df, model=model, split="train")
        evaluate_model(df=test_df, model=model, split="test")

        logger.info("All evaluation steps have been completed successfully!")


if __name__ == "__main__":
    main()