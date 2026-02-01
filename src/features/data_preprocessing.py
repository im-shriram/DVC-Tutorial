# Importing necessary libraries
import pandas as pd
import pathlib
import re
import json
from textblob import TextBlob
import emoji
import logging
from typing import Tuple, Dict

import nltk
# Downloading stopwords
nltk.download('stopwords')
# Downloading wordnet
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




# Loading Dataset
def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    logger = logging.getLogger(__name__)

    # Forming file paths
    train_path = data_dir / "interim" / "train.csv"
    test_path = data_dir / "interim" / "test.csv"
    chat_words = data_dir / "external" / "chat_words_dictonary.json"

    logger.info(f"Loading dataset from {data_dir / "interim"}")

    # Loading datasets
    train_df = pd.read_csv(filepath_or_buffer=train_path)
    test_df = pd.read_csv(filepath_or_buffer=test_path)

    # Loading chat word dictonary
    with open(file=chat_words, mode='r') as f:
        chat_words_map = json.load(f)

    # Returning datasets
    return (train_df, test_df, chat_words_map)

# Preprocessing
def preprocessing(df: pd.DataFrame, chat_words_map: Dict[str, str]):
    logger = logging.getLogger(__name__)
    
    # Lowercasing
    logger.debug("Lowercasing the dataset")
    df['content'] = df['content'].str.lower()

    # Removing HTML Tags
    logger.debug("Removing HTML Tags from the dataset")
    html_pattern = re.compile('<.*?>')
    df["content"] = df["content"].apply(lambda tweet: re.sub(html_pattern, '', tweet))

    # Removing Links
    logger.debug("Removing links and URL's from the dataset")
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    df["content"] = df["content"].apply(lambda tweet: re.sub(url_pattern, r'', tweet))

    # Remove Puncutations
    logger.debug("Removing puncutations from the dataset")
    punctuation_pattern = r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\\]^_`{|}~""")
    df["content"] = df["content"].apply(lambda tweet: re.sub(punctuation_pattern, ' ', tweet))

    df["content"] = df["content"].str.replace('؛', "", )

    # remove extra whitespace
    logger.debug("Removing extra white spaces from the dataset")
    df["content"] = df["content"].apply(lambda tweet: re.sub(r'\s+', ' ', tweet))
    df["content"] = df["content"].apply(lambda tweet: " ".join(tweet.split()).strip()) 

    # Chat Word Treatments - "really ill atm" -> atm (at this moment)
    logger.debug("Replacing the chat words with there full forms in the dataset")
    def handle_chat_word(tweet):
        tweet_words = tweet.split()
        for idx, word in enumerate(tweet_words):
            if word in chat_words_map.keys():
                tweet_words[idx] = chat_words_map[word]
            else: word
        return " ".join(tweet_words)
    df["content"] = df["content"].apply(lambda tweet: handle_chat_word(tweet=tweet))

    # Removing stop words
    logger.debug("Removing the stop words from the dataset")
    stop_words = set(stopwords.words("english"))
    df["content"] = df["content"].apply(lambda tweet: " ".join([word for word in str(tweet).split() if word not in stop_words]))

    # Handling emojis
    logger.debug("Replacing emojis with there textual representation from the dataset")
    df["content"] = df["content"].apply(lambda tweet: emoji.demojize(tweet))

    # Spelling Corrections
    logger.debug("Correcting the spellings in the tweets")
    # df["content"] = df["content"].apply(lambda tweet: TextBlob(tweet).correct())

    # Lemmatizetion
    logger.debug("Lemmatizing the words in the tweets from the dataset")
    lemmatizer= WordNetLemmatizer()
    df["content"] = df["content"].apply(lambda tweet: " ".join([lemmatizer.lemmatize(word) for word in tweet.split()]))

    # Return the precessed dataset
    return df

# Save the dataset
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: str) -> None:
    logger = logging.getLogger(__name__)

    # Forming the save path
    save_path = save_dir / "processed"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving the preprocessed datasets to {save_path}")

    # Saving the datasets
    train_df.to_csv(path_or_buf=save_path / "train.csv", index=False)
    test_df.to_csv(path_or_buf=save_path / "test.csv", index=False)

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
    # Forming Logger
    logger = form_logger()

    logger.info("Starting data preprocessing pipeline")

    # Forming directory paths
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    logger.info(f"Working directory: {home_dir}")

    # Loading Data
    train_df, test_df, chat_words_map = load_data(data_dir=data_dir)

    # Preprocessing Data
    logger.info(f"Started preprocessing train dataset")
    train_df = preprocessing(df=train_df, chat_words_map=chat_words_map)

    logger.info(f"Started preprocessing test dataset")
    test_df = preprocessing(df=test_df, chat_words_map=chat_words_map)

    # Saving the preprocessed data
    save_data(train_df=train_df, test_df=test_df, save_dir=data_dir)

    logger.info("Data preprocessing completed successfully")
if __name__ == "__main__":
    main()