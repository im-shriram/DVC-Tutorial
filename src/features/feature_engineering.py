# Importing necessary libraries
import pandas as pd
from typing import Tuple, Dict, Any
import logging
import yaml
import pathlib
import joblib
from sklearn.feature_extraction.text import CountVectorizer


# Loading Dataset
def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger = logging.getLogger(__name__)

    # Forming file paths
    train_path = data_dir / "processed" / "train.csv"
    test_path = data_dir / "processed" / "test.csv"

    logger.info(f"Loading dataset from {data_dir / "processed"}")

    # Loading datasets
    train_df = pd.read_csv(filepath_or_buffer=train_path)
    test_df = pd.read_csv(filepath_or_buffer=test_path)

    # Returning datasets
    return (train_df.dropna(), test_df.dropna())

# Loading Parameters
def load_params(params_path: pathlib.Path) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading parameters from {params_path}")
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Check if 'make_dataset' key exists in params
        if 'feature_engineering' not in params:
            logger.error("'feature_engineering' section not found in parameters file")
            raise KeyError("'feature_engineering' section not found in parameters file")
        
        logger.debug(f"Parameters loaded: {params['feature_engineering']}")
        return params['feature_engineering']

# Training and saving feature encoder
def train_vectorizer(train_df: pd.DataFrame, max_features: int, save_path: str) -> CountVectorizer:
    logger = logging.getLogger(__name__)
    logger.info(msg="Training the `Bag-of-Word` vectorizer")

    save_path = save_path / "vectorizers"
    save_path.mkdir(parents=True, exist_ok=True)

    # Forming and training the BOW vectorizer
    vectorizer = CountVectorizer(max_features=max_features)
    vectorizer.fit(train_df["content"].tolist())

    # Saving the vectorizer
    logger.info(msg="Saving the `Bag-of-Word` vectorizer")
    joblib.dump(value=vectorizer, filename=save_path / "bow.joblib")
    return vectorizer

# Transforming feature encoder
def encoding_feature(df: pd.DataFrame, vectorizer: CountVectorizer) -> pd.DataFrame:
    # Transform the entire content column at once instead of applying row by row
    content_transformed = vectorizer.transform(df['content'].values)
 
    # Convert sparse matrix to dense array
    df = pd.concat(objs=[df, pd.DataFrame(content_transformed.toarray())], axis=1)
    return df

# Save the dataset
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: str) -> None:
    logger = logging.getLogger(__name__)

    # Forming the save path
    save_path = save_dir / "features"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving the transformed datasets to {save_path}")

    # Saving the datasets
    train_df.to_csv(path_or_buf=save_path / "train.csv", index=False)
    test_df.to_csv(path_or_buf=save_path / "test.csv", index=False)

# Forming logger
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


# main function
def main() -> None:
    # Forming logger
    logger = form_logger()
    logger.info(msg="Started feature engineering pipeline")

    # Forming directory paths
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    params_path = home_dir / "params.yaml"
    models_path = home_dir / "models" # The place where we save all the trainable artifacts
    logger.info(f"Working directory: {home_dir}")

    # Loading data
    train_df, test_df = load_data(data_dir=data_dir)

    # Loading parameters
    params = load_params(params_path=params_path)

    # Training encoder
    vectorizer = train_vectorizer(train_df=train_df, max_features=params["bow_max_features"], save_path=models_path)

    # Transforming Features
    logger.info(msg="Encoding train dataset")
    train_df_encoded = encoding_feature(df=train_df, vectorizer=vectorizer)

    logger.info(msg="Encoding test dataset")
    test_df_encoded = encoding_feature(df=test_df, vectorizer=vectorizer)

    # Saving the transformed data
    save_data(train_df=train_df_encoded, test_df=test_df_encoded, save_dir=data_dir)

    logger.info("Feature Engineering completed successfully")

if __name__ == "__main__":
    main()