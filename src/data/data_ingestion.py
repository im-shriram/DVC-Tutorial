# Step 1: Import all required libraries for data manipulation, logging, and system path handling
import pandas as pd
import logging
import pathlib
import yaml
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


# Function to load project configuration parameters from a YAML file
def load_params(params_path: pathlib.Path) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Opening and reading configuration details from: {params_path}")
    
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Verify if the 'data_ingestion' section exists in the configuration file
        if 'data_ingestion' not in params:
            logger.error("The essential 'data_ingestion' section is missing from the configuration file!")
            raise KeyError("'data_ingestion' section not found in parameters file")
        
        logger.debug(f"Configuration for data ingestion successfully loaded: {params['data_ingestion']}")
        return params['data_ingestion']

# Function to load the raw dataset from the data directory
def load_data(data_dir: pathlib.Path) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    
    # Define the exact path to the raw input CSV file
    file_path = data_dir / "raw" / "train.csv"
    logger.info(f"Accessing the raw dataset file located at: {file_path}")
    
    with open(file=file_path, mode='r') as f:
        # Read the CSV content into a pandas DataFrame
        df = pd.read_csv(filepath_or_buffer=f)
        logger.debug(f"Dataset successfully loaded. Total rows/columns found: {df.shape}")
        return df
    
# Function to split the primary dataset into training and testing sets
def split_data(dataset: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger = logging.getLogger(__name__)
    
    # Inform about the percentage of data being reserved for testing
    test_ratio = params['test_size']
    logger.info(f"Splitting dataset: Reserving {test_ratio*100}% of samples for the testing set")
    
    # Perform the split using stratification to ensure balanced distribution of 'sentiment' labels
    train_df, test_df = train_test_split(dataset, 
                                        test_size=params['test_size'],
                                        stratify=dataset["sentiment"],
                                        random_state=42)
    
    logger.debug(f"Data split complete. Train set size: {train_df.shape}, Test set size: {test_df.shape}")
    return (train_df, test_df)

# Function to save the split datasets into a specific storage folder
def store_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: pathlib.Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing to store the processed datasets in the folder: {save_path}")
    
    # Automatically create the destination directory and any parent folders if they don't exist
    save_path.mkdir(parents=True, exist_ok=True)

    # Define the filenames for the training and testing sets
    train_path = save_path / "train.csv"
    test_path = save_path / "test.csv"
    
    # Save the DataFrames to CSV files without writing the row index numbers
    train_df.to_csv(path_or_buf=train_path, index=False)
    test_df.to_csv(path_or_buf=test_path, index=False)
    
    logger.debug(f"Training data successfully saved to: {train_path}")
    logger.debug(f"Testing data successfully saved to: {test_path}")

# Function to configure the global logging system for terminal output
def form_logger() -> logging.Logger:
    # Access the root logger to catch all messages throughout the application
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    # Create a console handler to print log messages to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Set a standardized format: [Timestamp] - [Logger Name] - [Log Level] - [Message]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Attach the handler only if one doesn't already exist to avoid message duplication
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


# Main entry point for the data ingestion and splitting process
def main() -> None:
    # Setup the logger to track the execution progress
    logger = form_logger()
    logger.info("Initializing the data ingestion and splitting pipeline")

    try:
        # Identify the root directory of the project and form paths for data and parameters
        home_dir = pathlib.Path(__file__).parent.parent.parent
        data_dir = home_dir / "data"
        params_path = home_dir / "params.yaml"
        
        logger.info(f"Project home directory detected as: {home_dir}")

        # Step 1: Load the configuration settings
        logger.info("Step 1: Loading ingestion parameters from configuration file")
        params = load_params(params_path=params_path)

        # Step 2: Access and load the raw dataset
        logger.info("Step 2: Accessing and loading the raw input data")
        dataset = load_data(data_dir=data_dir)

        # Step 3: Divide the data into training and testing portions
        logger.info("Step 3: Dividing the dataset into training and testing sets")
        train_df, test_df = split_data(dataset=dataset, params=params)

        # Step 4: Write the split datasets to the 'interim' data folder
        logger.info("Step 4: Writing the split datasets to the storage folder for downstream processing")
        save_path = data_dir / "interim"
        store_data(train_df=train_df, test_df=test_df, save_path=save_path)
        
        logger.info("Data ingestion and splitting pipeline has finished successfully!")
        
    except Exception as e:
        # Log any errors that occur during execution with detailed traceback information
        logger.error(f"A critical error occurred during data processing: {str(e)}", exc_info=True)
        raise

# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()