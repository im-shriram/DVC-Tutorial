import unittest
import joblib
import mlflow
import dagshub
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()
        # Set up MLflow tracking URI
        """
            mlflow.set_tracking_uri(uri="https://dagshub.com/Shriram-Vibhute/Emotion-Detection-MLOps-Practices.mlflow")
            dagshub.init(repo_owner='Shriram-Vibhute', repo_name='Emotion-Detection-MLOps-Practices', mlflow=True)
        """

        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Shriram-Vibhute"
        repo_name = "Emotion-Detection-MLOps-Practices"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "bagging_classifier"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        cls.vectorizer = joblib.load(open('models/vectorizers/bow.joblib', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/features/test.csv').dropna()

    @staticmethod
    def get_latest_model_version(model_name, stage="None"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = pd.DataFrame(
            {
                "content": ["hi how are you"], # Didnt include anything which requires preprocessing
                "sentiment": [0],
            }
        )
        input_data = self.vectorizer.transform(input_text["content"].values)
        df = pd.concat(objs=[input_text, pd.DataFrame(input_data.toarray())], axis=1).drop(columns=["content", "sentiment"])
        df.columns = df.columns.astype(str)

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(df)

        # Verify the input shape
        self.assertEqual(input_data.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.drop(columns=["sentiment", "content"])
        y_holdout = self.holdout_data["sentiment"]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted')
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted')
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted')

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.50
        expected_precision = 0.50
        expected_recall = 0.50
        expected_f1 = 0.50

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()