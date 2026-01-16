import joblib
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class MLService:
    
    def __init__(self, model_path: str, encoder_path: str):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.model = None
        self.label_encoder = None
        
    def load_artifacts(self):
        try:
            logger.info(f"Loading ML model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Loading label encoder from {self.encoder_path}")
            self.label_encoder = joblib.load(self.encoder_path)
            
            logger.info("ML artifacts loaded successfully")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Label classes: {self.label_encoder.classes_}")
            
        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}")
            raise
    
    def predict(self, ph: float, tds: float, turbidity: float, 
                temperature: float) -> Tuple[str, float]:
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("ML artifacts not loaded. Call load_artifacts() first.")
        
        features = np.array([[ph, tds, turbidity, temperature]])
        
        logger.info(f"Making prediction for features: {features}")
        
        prediction_encoded = self.model.predict(features)[0]
        
        prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probability (confidence) if available
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(max(probabilities))  # Confidence of predicted class
            logger.info(f"Prediction probabilities: {probabilities}")
        
        logger.info(f"Prediction: {prediction_label} (confidence: {confidence})")
        
        return prediction_label, confidence


# Global ML service instance (initialized in main.py)
ml_service: MLService = None
