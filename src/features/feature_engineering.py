# Step 1: Import necessary libraries for data manipulation, logging, and vectorization
import pandas as pd
from typing import Tuple, Dict, Any
import logging
import yaml
import pathlib
import joblib
from sklearn.feature_extraction.text import CountVectorizer


# Function to load the cleaned datasets from the 'processed' folder
def load_data(data_dir: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger = logging.getLogger(__name__)

    # Form the directory paths for the preprocessed training and testing files
    train_path = data_dir / "processed" / "train.csv"
    test_path = data_dir / "processed" / "test.csv"

    logger.info(f"Loading preprocessed datasets from the directory: {data_dir / 'processed'}")

    # Read the data and ensure no missing values are present before feature creation
    train_df = pd.read_csv(filepath_or_buffer=train_path)
    test_df = pd.read_csv(filepath_or_buffer=test_path)

    # Return the DataFrames with any NaN values removed
    return (train_df.dropna(), test_df.dropna())

# Function to load feature engineering parameters from the configuration file
def load_params(params_path: pathlib.Path) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Accessing configuration parameters from: {params_path}")
    
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Check if the 'feature_engineering' section is defined in the YAML file
        if 'feature_engineering' not in params:
            logger.error("The required 'feature_engineering' configuration section is missing!")
            raise KeyError("'feature_engineering' section not found in parameters file")
        
        logger.debug(f"Feature engineering parameters successfully loaded: {params['feature_engineering']}")
        return params['feature_engineering']

# Function to train and save the Bag-of-Words (BOW) text vectorizer
def train_vectorizer(train_df: pd.DataFrame, max_features: int, save_path: pathlib.Path) -> CountVectorizer:
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing training for the 'Bag-of-Words' vectorizer with a limit of {max_features} features")

    # Create the directory where the vectorizer model will be saved
    save_path = save_path / "vectorizers"
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize the vectorizer with the specified number of top frequent words
    vectorizer = CountVectorizer(max_features=max_features)
    # Learn the vocabulary from the training data
    vectorizer.fit(train_df["content"].tolist())

    # Save the trained vectorizer object to the disk for use in future predictions
    logger.info(f"Saving the trained 'Bag-of-Words' model to: {save_path / 'bow.joblib'}")
    joblib.dump(value=vectorizer, filename=save_path / "bow.joblib")
    
    return vectorizer

# Function to transform text content into numerical features using the trained vectorizer
def encoding_feature(df: pd.DataFrame, vectorizer: CountVectorizer) -> pd.DataFrame:
    # Convert the 'content' column into a numerical matrix (Count matrix)
    # We transform all rows at once to optimize performance
    content_transformed = vectorizer.transform(df['content'].values)
 
    # Combine the original DataFrame with the newly created numerical feature columns
    # We convert the sparse matrix back to a standard format (dense array) for the final DataFrame
    df = pd.concat(objs=[df, pd.DataFrame(content_transformed.toarray())], axis=1)
    
    return df

# Function to save the final engineered datasets into the 'features' folder
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: pathlib.Path) -> None:
    logger = logging.getLogger(__name__)

    # Form the directory path for the final feature files
    save_path = save_dir / "features"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing final engineered datasets to: {save_path}")

    # Save the training and testing sets as CSV files for the model training stage
    train_df.to_csv(path_or_buf=save_path / "train.csv", index=False)
    test_df.to_csv(path_or_buf=save_path / "test.csv", index=False)
    
    logger.debug("Successfully saved engineered train.csv and test.csv")

# Function to configure the system logger for terminal reporting
def form_logger() -> logging.Logger:
    logger = logging.getLogger()
    # Set to DEBUG to monitor vocabulary size and transformation details
    logger.setLevel(level=logging.DEBUG)

    # Setup the console handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # Standard log layout: [Timestamp] - [Module Name] - [Level] - [Message]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Ensure handlers are not added multiple times in the same session
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# Main entry point for the feature engineering pipeline
def main() -> None:
    # Setup the logger for tracking process execution
    logger = form_logger()
    logger.info("The feature engineering pipeline has officially started")

    # Resolve necessary directory paths for data and model artifacts
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    params_path = home_dir / "params.yaml"
    models_path = home_dir / "models" 
    
    logger.info(f"Project base path identified at: {home_dir}")

    # Step 1: Load the cleaned datasets
    logger.info("Step 1: Loading preprocessed training and testing data")
    train_df, test_df = load_data(data_dir=data_dir)

    # Step 2: Load the engineering parameters (like max features)
    logger.info("Step 2: Retrieving feature limits from the configuration file")
    params = load_params(params_path=params_path)

    # Step 3: Train the numerical encoder (Vectorizer)
    logger.info("Step 3: Training the numerical text encoder on the training dataset")
    vectorizer = train_vectorizer(train_df=train_df, max_features=params["bow_max_features"], save_path=models_path)

    # Step 4: Transform both datasets into numerical features
    logger.info("Step 4: Applying transformations to encode train and test datasets")
    train_df_encoded = encoding_feature(df=train_df, vectorizer=vectorizer)
    test_df_encoded = encoding_feature(df=test_df, vectorizer=vectorizer)

    # Step 5: Save the final datasets for model training
    logger.info("Step 5: Saving the finalized numerical feature sets for the modeling stage")
    save_data(train_df=train_df_encoded, test_df=test_df_encoded, save_dir=data_dir)

    logger.info("Feature engineering stage has successfully completed!")

# Execute the feature engineering script
if __name__ == "__main__":
    main()