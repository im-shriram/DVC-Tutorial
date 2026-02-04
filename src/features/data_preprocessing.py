# Step 1: Import necessary libraries for text processing, regular expressions, and logging
import pandas as pd
import pathlib
import re
import json
from textblob import TextBlob
import emoji
import logging
from typing import Tuple, Dict

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Function to load the split datasets and the chat words dictionary
def load_data(data_dir: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    logger = logging.getLogger(__name__)

    # Form the directory paths for interim data and the external chat words dictionary
    train_path = data_dir / "interim" / "train.csv"
    test_path = data_dir / "interim" / "test.csv"
    chat_words_path = data_dir / "external" / "chat_words_dictonary.json"

    logger.info(f"Loading split datasets for cleaning from: {data_dir / 'interim'}")

    # Load the training and testing sets into DataFrames
    train_df = pd.read_csv(filepath_or_buffer=train_path)
    test_df = pd.read_csv(filepath_or_buffer=test_path)

    # Load the mapping dictionary used to expand internet slang (e.g., 'lol' to 'laughing out loud')
    logger.info(f"Loading chat words expansion dictionary from: {chat_words_path}")
    with open(file=chat_words_path, mode='r') as f:
        chat_words_map = json.load(f)

    # Return the datasets and the mapping dictionary
    return (train_df, test_df, chat_words_map)

# Function to perform various text cleaning and normalization steps on the dataset
def preprocessing(df: pd.DataFrame, chat_words_map: Dict[str, str]) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    
    # Step 1: Convert all text to lowercase to ensure consistency
    logger.debug("Lowercasing all text content to remove case sensitivity")
    df['content'] = df['content'].str.lower()

    # Step 2: Remove any HTML tags that might be present in the text (e.g., <br>)
    logger.debug("Removing HTML tags (like <div> or <br>) from the content")
    html_pattern = re.compile('<.*?>')
    df["content"] = df["content"].apply(lambda text: re.sub(html_pattern, '', str(text)))

    # Step 3: Remove website URLs and links
    logger.debug("Searching for and removing URLs and web links from the content")
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    df["content"] = df["content"].apply(lambda text: re.sub(url_pattern, r'', str(text)))

    # Step 4: Remove punctuation marks to focus on the words themselves
    logger.debug("Removing punctuation marks while preserving whitespace")
    punctuation_pattern = r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\\]^_`{|}~""")
    df["content"] = df["content"].apply(lambda text: re.sub(punctuation_pattern, ' ', str(text)))
    df["content"] = df["content"].str.replace('؛', "", ) # Remove specific non-standard punctuation

    # Step 5: Clean up extra spaces caused by the previous removal steps
    logger.debug("Consolidating multiple whitespaces into a single space")
    df["content"] = df["content"].apply(lambda text: re.sub(r'\s+', ' ', str(text)))
    df["content"] = df["content"].apply(lambda text: " ".join(text.split()).strip()) 

    # Step 6: Expand chat slang and abbreviations using the loaded dictionary
    logger.debug("Expanding chat abbreviations and slang (e.g., 'atm' -> 'at this moment')")
    def handle_chat_word(text):
        words = text.split()
        for idx, word in enumerate(words):
            if word in chat_words_map:
                words[idx] = chat_words_map[word]
        return " ".join(words)
    df["content"] = df["content"].apply(lambda text: handle_chat_word(text=str(text)))

    # Step 7: Remove common 'stop words' (like 'the', 'is', 'a') which carry little meaning
    logger.debug("Removing common English stop words (like 'the', 'is', 'and')")
    stop_words = set(stopwords.words("english"))
    df["content"] = df["content"].apply(lambda text: " ".join([word for word in str(text).split() if word not in stop_words]))

    # Step 8: Convert emojis into their textual description (e.g., :smile:)
    logger.debug("Converting emojis into their descriptive text versions")
    df["content"] = df["content"].apply(lambda text: emoji.demojize(str(text)))

    # Step 9: Use Lemmatization to convert words back to their base or root form
    logger.debug("Applying lemmatization to convert words to their root form (e.g., 'running' -> 'run')")
    lemmatizer = WordNetLemmatizer()
    df["content"] = df["content"].apply(lambda text: " ".join([lemmatizer.lemmatize(word) for word in str(text).split()]))

    # Return the fully cleaned and normalized dataset
    return df

# Function to save the preprocessed datasets into the 'processed' folder
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: pathlib.Path) -> None:
    logger = logging.getLogger(__name__)

    # Create the full path for the processed data folder
    save_path = save_dir / "processed"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing the final preprocessed datasets to: {save_path}")

    # Write the cleaned datasets to CSV files without the index column
    train_df.to_csv(path_or_buf=save_path / "train.csv", index=False)
    test_df.to_csv(path_or_buf=save_path / "test.csv", index=False)
    logger.debug("Successfully saved processed train.csv and test.csv")

# Function to initialize the system logger for terminal reporting
def form_logger() -> logging.Logger:
    logger = logging.getLogger()
    # Set to DEBUG to capture detailed logs during the cleaning process
    logger.setLevel(level=logging.DEBUG)

    # Setup the terminal output handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # Standard log format: [Time] - [Module Name] - [Level] - [Message]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Prevent duplicating logs if script is run multiple times in same session
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# Main entry point for the data cleaning (preprocessing) pipeline
def main() -> None:
    # Initialize the logger
    logger = form_logger()
    logger.info("Initializing the data cleaning and preprocessing pipeline")

    # Resolve necessary directory paths
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    logger.info(f"System workspace identified at: {home_dir}")

    # Step 1: Load the raw data and reference dictionary
    logger.info("Step 1: Loading split datasets and chat words dictionary")
    train_df, test_df, chat_words_map = load_data(data_dir=data_dir)

    # Step 2: Clean the training dataset
    logger.info("Step 2: Performing comprehensive cleaning on the training dataset")
    train_df = preprocessing(df=train_df, chat_words_map=chat_words_map)

    # Step 3: Clean the testing dataset
    logger.info("Step 3: Performing comprehensive cleaning on the testing dataset")
    test_df = preprocessing(df=test_df, chat_words_map=chat_words_map)

    # Step 4: Save the cleaned data for feature engineering
    logger.info("Step 4: Saving prepared datasets for the feature engineering stage")
    save_data(train_df=train_df, test_df=test_df, save_dir=data_dir)

    logger.info("Data preprocessing pipeline has successfully finished!")

# Run the pipeline
if __name__ == "__main__":
    main()