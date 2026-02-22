"""
Sales Forecasting Model Training Script
Trains a time-series forecasting model for sales prediction using Prophet.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.pyfunc
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:admin123@localhost:5432/enterprise_db')
MLFLOW_TRACKING_URI = './mlruns'
MODEL_OUTPUT_PATH = './ml_engine/models/forecast_model.pkl'

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("sales_forecasting")


def load_sales_data():
    """Load sales data from PostgreSQL database."""
    print("📊 Loading sales data from database...")
    
    engine = create_engine(DATABASE_URL)
    
    query = """
        SELECT 
            sale_date,
            SUM(total_amount) as daily_revenue,
            COUNT(*) as transaction_count,
            AVG(total_amount) as avg_transaction_value
        FROM sales
        GROUP BY sale_date
        ORDER BY sale_date
    """
    
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    print(f"✅ Loaded {len(df)} days of sales data")
    print(f"   Date range: {df['sale_date'].min()} to {df['sale_date'].max()}")
    print(f"   Total revenue: ${df['daily_revenue'].sum():,.2f}")
    
    return df


def prepare_prophet_data(df):
    """Prepare data for Prophet model."""
    print("🔧 Preparing data for Prophet...")
    
    # Prophet requires columns named 'ds' (date) and 'y' (value)
    prophet_df = df[['sale_date', 'daily_revenue']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Add additional regressors
    prophet_df['transaction_count'] = df['transaction_count'].values
    prophet_df['avg_transaction_value'] = df['avg_transaction_value'].values
    
    print(f"✅ Prepared {len(prophet_df)} data points")
    
    return prophet_df


def train_forecast_model(df):
    """Train Prophet forecasting model."""
    print("\n🤖 Training Prophet forecasting model...")
    
    # Split data (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    print(f"   Training set: {len(train_df)} days")
    print(f"   Test set: {len(test_df)} days")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Model parameters
        params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Initialize and train model
        model = Prophet(**params)
        
        # Add additional regressors
        model.add_regressor('transaction_count')
        model.add_regressor('avg_transaction_value')
        
        # Fit model
        model.fit(train_df)
        
        # Make predictions on test set
        forecast = model.predict(test_df[['ds', 'transaction_count', 'avg_transaction_value']])
        
        # Calculate metrics
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape", mape)
        
        print("\n📊 Model Performance:")
        print(f"   MAE:   ${mae:,.2f}")
        print(f"   RMSE:  ${rmse:,.2f}")
        print(f"   R²:    {r2:.4f}")
        print(f"   MAPE:  {mape:.2f}%")
        
        # Generate future forecast (next 30 days)
        future_dates = model.make_future_dataframe(periods=30)
        
        # For future dates, use average values from recent data
        recent_avg_transactions = train_df['transaction_count'].tail(30).mean()
        recent_avg_value = train_df['avg_transaction_value'].tail(30).mean()
        
        future_dates['transaction_count'] = recent_avg_transactions
        future_dates['avg_transaction_value'] = recent_avg_value
        
        future_forecast = model.predict(future_dates)
        
        # Get next month's forecast
        next_month_forecast = future_forecast.tail(30)
        predicted_revenue = next_month_forecast['yhat'].sum()
        
        print(f"\n📈 Next 30 Days Forecast:")
        print(f"   Predicted Revenue: ${predicted_revenue:,.2f}")
        print(f"   Daily Average: ${predicted_revenue/30:,.2f}")
        
        mlflow.log_metric("next_month_forecast", predicted_revenue)
        
        # Save model locally
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        model_data = {
            'model': model,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'mape': mape
            },
            'next_month_forecast': predicted_revenue,
            'recent_avg_transactions': recent_avg_transactions,
            'recent_avg_value': recent_avg_value
        }
        joblib.dump(model_data, MODEL_OUTPUT_PATH)
        
        print(f"\n✅ Model saved to {MODEL_OUTPUT_PATH}")
        
        return model, forecast


def main():
    """Main training pipeline."""
    print("="*60)
    print("🚀 SALES FORECASTING MODEL TRAINING")
    print("="*60)
    
    try:
        # Load data
        df = load_sales_data()
        
        # Prepare data
        prophet_df = prepare_prophet_data(df)
        
        # Train model
        model, forecast = train_forecast_model(prophet_df)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
