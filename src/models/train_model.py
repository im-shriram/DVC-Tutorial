# Step 1: Import required libraries for logging, model serialization, data handling, and machine learning
import logging
import joblib
import pandas as pd
import pathlib
from typing import Dict, Any
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


# Function to load the engineered training dataset from the features directory
def load_data(data_dir: pathlib.Path) -> pd.DataFrame:
    logger = logging.getLogger(__name__)

    # Form the directory path where numerical features are stored
    train_path = data_dir / "features" / "train.csv"
    logger.info(f"Loading numerical training features from: {data_dir / 'features'}")

    # Load the training data into a pandas DataFrame
    train_df = pd.read_csv(filepath_or_buffer=train_path)

    # Return the training data while ensuring no missing values are passed to the model
    logger.debug(f"Training dataset loaded with shape: {train_df.shape}")
    return train_df.dropna()

# Function to load and organize training hyperparameters from the configuration file
def load_params(params_path: pathlib.Path) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Opening and reading model hyperparameters from: {params_path}")
    
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Verify if the 'model_training' configuration section is present
        if 'model_training' not in params:
            logger.error("The essential 'model_training' section is missing from the params.yaml file!")
            raise KeyError("model_training section not found in parameters file")
        
        # Organize the parameters into two groups: one for the base estimator and one for the bagging ensemble
        processed_params = {
            "bagging": dict(),
            "estimator": dict()
        }
        
        # Map parameters starting with 'bagging_' to the bagging ensemble configuration
        for key, value in params['model_training']["bagging"].items():
            if str(key).startswith("bagging"):
                processed_params["bagging"][str(key).removeprefix("bagging_")] = value
            else:
                processed_params[key] = value
        
        # Map parameters starting with 'estimator_' to the base decision tree configuration
        for key, value in params['model_training']["estimator"].items():
            if str(key).startswith("estimator"):
                processed_params["estimator"][str(key).removeprefix("estimator_")] = value
            else:
                processed_params[key] = value

        logger.info("Configuration parameters successfully organized for model building")
        return processed_params

# Function to construct and train the Bagging Classifier model
def train_model(df: pd.DataFrame, params: Dict[str, Any]) -> BaggingClassifier:
    logger = logging.getLogger(__name__)
    logger.info("Initializing the process to build and train the 'Bagging Classifier' model")

    # Separate the target label ('sentiment') and helper columns from the numerical features (X)
    X_train = df.drop(columns=["content", "sentiment"])
    y_train = df["sentiment"]

    # Construct the ensemble model using a Decision Tree as the base estimator
    # The '**' operator unpacks the parameter dictionaries into the class constructors
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(**params["estimator"]),
        **params["bagging"]
    )

    # Fit the model on the training data to learn patterns
    logger.info(f"Starting model fitting on {len(X_train)} samples...")
    model.fit(X=X_train, y=y_train)
    logger.info("Model fitting has finished successfully")
    
    return model

# Function to save the trained model object for future use
def save_model(model: BaggingClassifier, save_dir: pathlib.Path) -> None:
    logger = logging.getLogger(__name__)

    # Create the directory for saving the trained models
    save_path = save_dir / "models"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save the model object to a joblib file for efficient storage and loading
    model_filename = "bagging_classifier.joblib"
    logger.info(f"Writing the final trained model to: {save_path / model_filename}")
    joblib.dump(value=model, filename=save_path / model_filename)
    logger.debug("Successfully saved the model artifact")

# Function to configure the system-wide logging for console output
def form_logger() -> logging.Logger:
    logger = logging.getLogger()
    # Log level set to DEBUG to track all training internal details
    logger.setLevel(level=logging.DEBUG)

    # Attach a terminal handler to print messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # Standard log layout: [Timestamp] - [Module] - [Level] - [Message]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Avoid duplicate log messages by checking if a handler is already present
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# Main entry point for the model training stage of the pipeline
def main() -> None:
    # Set up the logger
    logger = form_logger()
    logger.info("The model training pipeline has officially started")

    # Define necessary directory paths for data, models, and execution parameters
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    params_path = home_dir / "params.yaml"
    models_path = home_dir / "models"
    
    logger.info(f"Base project path identified at: {home_dir}")

    # Step 1: Load numerical features for training
    logger.info("Step 1: Loading engineered numerical features for model input")
    train_df = load_data(data_dir=data_dir)

    # Step 2: Retrieve hyperparameters for model configuration
    logger.info("Step 2: Loading model hyperparameters from project configuration")
    params = load_params(params_path=params_path)

    # Step 3: Build the model and perform training (fitting)
    logger.info("Step 3: Constructing and training the Bagging Classifier ensemble")
    model = train_model(df=train_df, params=params)

    # Step 4: Save the finalized model artifact for evaluation and deployment
    logger.info("Step 4: Saving the finalized model object to the internal storage")
    save_model(model=model, save_dir=models_path)

    logger.info("The model training pipeline has successfully completed!")

# Execute the training pipeline script
if __name__ == "__main__":
    main()