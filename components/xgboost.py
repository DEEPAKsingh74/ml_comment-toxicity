import joblib
import logging

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def load_model(self, version='v1'):
        try:
            # Dynamically construct file paths based on the version
            model_path = f"./models/xgb/xgboost_model_{version}.joblib"
            vectorizer_path = f"./models/vectorizer/tfidf_vectorizer_{version}.joblib"
            
            # Load the trained XGBoost model
            self.model = joblib.load(model_path)
            logging.info(f"Model loaded successfully for version {version}.")
            
            # Verify if the model is loaded by checking its type or attributes
            if not hasattr(self.model, "predict"):
                raise ValueError("The model instance does not have a predict method. Ensure the model is loaded correctly.")
            
            # Load the pre-trained TF-IDF Vectorizer
            self.vectorizer = joblib.load(vectorizer_path)
            logging.info(f"TF-IDF Vectorizer loaded successfully for version {version}.")
            
        except FileNotFoundError as e:
            logging.error(f"Model or Vectorizer file not found for version {version}: {e}")
            raise e  # Re-raise the exception after logging
            
        except Exception as e:
            logging.error(f"Error loading model or vectorizer for version {version}: {e}")
            raise e

    def prediction(self, comment):
        try:
            # Wrap the comment in a list
            comment = [comment]
            comment_vectorized = self.preprocess(comment)
            prediction = self.model.predict(comment_vectorized)
            return prediction
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e

    def preprocess(self, comment):
        try:
            # Transform the comment using the vectorizer
            comment_vectorized = self.vectorizer.transform(comment)
            return comment_vectorized
            
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise e
