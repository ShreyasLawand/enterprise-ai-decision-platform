"""
Model Inference Module
Loads trained models and provides prediction functions.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# Model paths
CHURN_MODEL_PATH = os.getenv('CHURN_MODEL_PATH', './ml_engine/models/churn_model.pkl')
FORECAST_MODEL_PATH = os.getenv('FORECAST_MODEL_PATH', './ml_engine/models/forecast_model.pkl')


class ChurnPredictor:
    """Churn prediction model wrapper."""
    
    def __init__(self):
        self.model_data = None
        self.model = None
        self.feature_columns = None
        self.metrics = None
    
    def load_model(self):
        """Load the trained churn model."""
        if not os.path.exists(CHURN_MODEL_PATH):
            raise FileNotFoundError(f"Churn model not found at {CHURN_MODEL_PATH}")
        
        self.model_data = joblib.load(CHURN_MODEL_PATH)
        self.model = self.model_data['model']
        self.feature_columns = self.model_data['feature_columns']
        self.metrics = self.model_data['metrics']
        
        print(f"✅ Churn model loaded successfully")
        print(f"   Model metrics: {self.metrics}")
    
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn probability for a customer.
        
        Args:
            customer_data: Dictionary containing customer features
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            self.load_model()
        
        # Prepare features
        features = pd.DataFrame([customer_data])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0  # Default value for missing features
        
        # Select only required features in correct order
        X = features[self.feature_columns]
        
        # Make prediction
        churn_probability = self.model.predict_proba(X)[0][1]
        churn_prediction = int(churn_probability > 0.5)
        
        # Determine risk level
        if churn_probability < 0.3:
            risk_level = "Low"
        elif churn_probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'churn_probability': float(churn_probability),
            'will_churn': bool(churn_prediction),
            'risk_level': risk_level,
            'model_metrics': self.metrics
        }


class SalesForecaster:
    """Sales forecasting model wrapper."""
    
    def __init__(self):
        self.model_data = None
        self.model = None
        self.metrics = None
    
    def load_model(self):
        """Load the trained forecast model."""
        if not os.path.exists(FORECAST_MODEL_PATH):
            raise FileNotFoundError(f"Forecast model not found at {FORECAST_MODEL_PATH}")
        
        self.model_data = joblib.load(FORECAST_MODEL_PATH)
        self.model = self.model_data['model']
        self.metrics = self.model_data['metrics']
        
        print(f"✅ Forecast model loaded successfully")
        print(f"   Model metrics: {self.metrics}")
    
    def forecast_next_month(self, days: int = 30) -> Dict[str, Any]:
        """
        Forecast sales for the next N days.
        
        Args:
            days: Number of days to forecast
        
        Returns:
            Dictionary with forecast results
        """
        if self.model is None:
            self.load_model()
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=days)
        
        # Add regressors with recent averages
        future['transaction_count'] = self.model_data['recent_avg_transactions']
        future['avg_transaction_value'] = self.model_data['recent_avg_value']
        
        # Make forecast
        forecast = self.model.predict(future)
        
        # Get only future predictions
        future_forecast = forecast.tail(days)
        
        total_forecast = future_forecast['yhat'].sum()
        daily_average = total_forecast / days
        
        return {
            'total_forecast': float(total_forecast),
            'daily_average': float(daily_average),
            'forecast_days': days,
            'forecast_data': future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
            'model_metrics': self.metrics
        }


# Global instances
churn_predictor = ChurnPredictor()
sales_forecaster = SalesForecaster()


def get_churn_prediction(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for churn prediction."""
    return churn_predictor.predict(customer_data)


def get_sales_forecast(days: int = 30) -> Dict[str, Any]:
    """Convenience function for sales forecasting."""
    return sales_forecaster.forecast_next_month(days)
