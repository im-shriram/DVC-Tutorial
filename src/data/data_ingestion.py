# Importing required libraries
import pandas as pd
import logging
import pathlib
import yaml
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


# Loading Parameters
def load_params(params_path: pathlib.Path) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading parameters from {params_path}")
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Check if 'data_ingestion' key exists in params
        if 'data_ingestion' not in params:
            logger.error("'data_ingestion' section not found in parameters file")
            raise KeyError("'data_ingestion' section not found in parameters file")
        
        logger.debug(f"Parameters loaded: {params['data_ingestion']}")
        return params['data_ingestion']

# Loading Dataset
def load_data(data_dir: pathlib.Path) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    file_path = data_dir / "raw" / "train.csv"
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file=file_path, mode='r') as f:
        df = pd.read_csv(filepath_or_buffer=f)
        logger.debug(f"Dataset loaded with shape: {df.shape}")
        return df
    
# Splitting Dataset
def split_data(dataset: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger = logging.getLogger(__name__)
    logger.info(f"Splitting dataset with test_size={params['test_size']}")
    
    train_df, test_df = train_test_split(dataset, 
                                        test_size=params['test_size'],
                                        stratify=dataset["sentiment"],
                                        random_state=42)
    
    logger.debug(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
    return (train_df, test_df)

# Storing Dataset
def store_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: pathlib.Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Storing datasets in {save_path}")
    
    # Create folder if not exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Saving the dataset
    train_path = save_path / "train.csv"
    test_path = save_path / "test.csv"
    
    train_df.to_csv(path_or_buf=train_path, index=False)
    test_df.to_csv(path_or_buf=test_path, index=False)
    
    logger.debug(f"Saved train dataset to {train_path}")
    logger.debug(f"Saved test dataset to {test_path}")

# Forming Logger
def form_logger() -> logging.Logger:
    logger = logging.getLogger() # If no name is specified, return the root logger. Thats why we are accessing the logger through - logging.getLogger(__name__) in every function
    logger.setLevel(logging.INFO)

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger if it doesn't already exist
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


# Main Function
def main() -> None:
    # Forming Logger
    logger = form_logger()
    logger.info("Starting data loading and splitting pipeline")

    try:
        # Forming paths for loading and storing the dataset
        home_dir = pathlib.Path(__file__).parent.parent.parent
        data_dir = home_dir / "data"
        params_path = home_dir / "params.yaml"
        
        logger.info(f"Working directory: {home_dir}")

        # Loading the parameters
        params = load_params(params_path=params_path)

        # Loading the Dataset
        dataset = load_data(data_dir=data_dir)

        # Splitting the data into Train and Test
        train_df, test_df = split_data(dataset=dataset, params=params)

        # Storing the data into interim folder
        save_path = data_dir / "interim"
        store_data(train_df=train_df, test_df=test_df, save_path=save_path)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()