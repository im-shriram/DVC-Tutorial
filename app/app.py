import pandas as pd
import json
import joblib
import mlflow
mlflow.set_tracking_uri(f'https://dagshub.com/Shriram-Vibhute/DVC-Tutorial.mlflow')
dagshub.init(repo_owner='Shriram-Vibhute', repo_name='DVC-Tutorial', mlflow=True)
from flask import Flask, request, render_template

# Preprocessing
from preprocessing import preprocessing
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
from preprocessing import encoding_feature
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
    app.run(debug=True, host="0.0.0.0", port=5000)