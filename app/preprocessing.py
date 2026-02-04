import pandas as pd
import re
import emoji
import logging
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer


# Downloading the stopwords and wordnet - Error while running the docker container
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


def encoding_feature(df: pd.DataFrame, vectorizer: CountVectorizer) -> pd.DataFrame:
    # Transform the entire content column at once instead of applying row by row
    content_transformed = vectorizer.transform(df['content'].values)
 
    # Convert sparse matrix to dense array
    df = pd.concat(objs=[df, pd.DataFrame(content_transformed.toarray())], axis=1)
    return df