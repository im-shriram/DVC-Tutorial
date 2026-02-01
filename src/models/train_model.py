# Importing necessary libraries
import logging
import joblib
import pandas as pd
import pathlib
from typing import Dict, Any
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Loading Data
def load_data(data_dir: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)

    # Forming file paths
    train_path = data_dir / "features" / "train.csv"
    logger.info(f"Loading dataset from {data_dir / "features"}")

    # Loading datasets
    train_df = pd.read_csv(filepath_or_buffer=train_path)

    # Returning datasets
    return train_df.dropna()

# Loading Parameters
def load_params(params_path: pathlib.Path) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading parameters from {params_path}")
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Check if 'make_dataset' key exists in params
        if 'model_training' not in params:
            logger.error("model_training section not found in parameters file")
            raise KeyError("model_training section not found in parameters file")
        
        # Preprocessing the params
        processed_params = {
            "bagging": dict(),
            "estimator": dict()
        }
        for key, value in params['model_training']["bagging"].items():
            if str(key).startswith("bagging"):
                processed_params["bagging"][str(key).removeprefix("bagging_")] = value
            else:
                processed_params[key] = value
        
        for key, value in params['model_training']["estimator"].items():
            if str(key).startswith("estimator"):
                processed_params["estimator"][str(key).removeprefix("estimator_")] = value
            else:
                processed_params[key] = value

        return processed_params

# Build and Train Random Forest model
def train_model(df: pd.DataFrame, params: Dict[str, str]) -> BaggingClassifier:
    logger = logging.getLogger(__name__)
    logger.info(msg="Building and Training `BaggingClasifier` Model")

    # Splitting the data
    X_train = df.drop(columns=["content", "sentiment"])
    y_train = df["sentiment"]

    # Building the model
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(**params["estimator"]),
        **params["bagging"]
    )

    # Training the model
    model.fit(X=X_train, y=y_train)
    return model

# Save the Model
def save_model(model: BaggingClassifier, save_dir: str) -> None:
    logger = logging.getLogger(__name__)

    save_path = save_dir / "models"
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(msg=f"Saving the `BaggingClasifier` Model at {save_path}")
    
    joblib.dump(value=model, filename=save_path / "bagging_classifier.joblib")

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
    logger.info(msg="Started model training pipeline")

    # Forming directory paths
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    params_path = home_dir / "params.yaml"
    models_path = home_dir / "models" # The place where we save all the trainable artifacts
    logger.info(f"Working directory: {home_dir}")

    # Loading data
    train_df = load_data(data_dir=data_dir)

    # Loading parameters
    params = load_params(params_path=params_path)

    # Building and Training model
    model = train_model(df=train_df, params=params)

    # Saving model
    save_model(model=model, save_dir=models_path)

    logger.info(msg="Model Training completed successfully")

if __name__ == "__main__":
    main()