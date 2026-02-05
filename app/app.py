# Importing necessary libraries
import pandas as pd
import json
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from flask import Flask, request, render_template

import mlflow
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    print("WARNING: DAGSHUB_PAT environment variable is not set!")
    print("If you are running in Docker, use: docker run -e DAGSHUB_PAT=your_token ...")
else:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Shriram-Vibhute"
    repo_name = "Emotion-Detection-MLOps-Practices"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

try:
    from .preprocessing import preprocessing, encoding_feature
except Exception as e:
    from app.preprocessing import preprocessing, encoding_feature
    
# Preprocessing
def normalize_text(text):
    # Since preprocessing function excepts the data in the form of dataframe
    df = pd.DataFrame(
        {
            "content": [text],
            "sentiment": [0] # dummy label
        }
    )
    df = preprocessing(df, json.load(open("data/external/chat_words_dictonary.json")))
    return df

# Feature Engineering
def encode_features(df):
    vectorizer = joblib.load('models/vectorizers/bow.joblib')
    df = encoding_feature(df=df, vectorizer=vectorizer)
    return df

# Model Loading from mlflow model registry
model_name = "bagging_classifier"
production_model_uri = f"models:/{model_name}@production"
model = mlflow.pyfunc.load_model(model_uri=production_model_uri)

# Model Serving
def predict_sentiment(text):
    df = normalize_text(text)
    df = encode_features(df).drop(columns=["content", "sentiment"])
    df.columns = df.columns.astype(str) # convert column names to string and not the values inside those columns
    result = model.predict(df)
    return result

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    result = predict_sentiment(text)
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) # NOTE: Need to add this 