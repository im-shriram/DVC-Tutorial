import pandas as pd
from typing import Tuple, Dict, Any
import logging
import yaml
import pathlib
import joblib
from sklearn.feature_extraction.text import CountVectorizer

def encoding_feature(df: pd.DataFrame, vectorizer: CountVectorizer) -> pd.DataFrame:
    # Transform the entire content column at once instead of applying row by row
    content_transformed = vectorizer.transform(df['content'].values)
 
    # Convert sparse matrix to dense array
    df = pd.concat(objs=[df, pd.DataFrame(content_transformed.toarray())], axis=1)
    return df