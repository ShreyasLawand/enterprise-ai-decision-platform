"""
Churn Prediction Model Training Script
Trains a Random Forest classifier to predict customer churn and logs to MLflow.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:admin123@localhost:5432/enterprise_db')
MLFLOW_TRACKING_URI = './mlruns'
MODEL_OUTPUT_PATH = './ml_engine/models/churn_model.pkl'

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("churn_prediction")


def load_data_from_db():
    """Load customer data from PostgreSQL database."""
    print("📊 Loading data from database...")
    
    engine = create_engine(DATABASE_URL)
    
    query = """
        SELECT 
            c.customer_id,
            c.subscription_tier,
            c.subscription_status,
            c.monthly_spend,
            c.total_lifetime_value,
            c.account_age_days,
            c.support_tickets_count,
            c.industry,
            c.country,
            (CURRENT_DATE - c.last_login_date) as days_since_last_login,
            COUNT(DISTINCT s.sale_id) as total_purchases,
            COALESCE(SUM(s.total_amount), 0) as total_purchase_amount,
            COUNT(DISTINCT st.ticket_id) as total_support_tickets,
            COUNT(DISTINCT CASE WHEN st.status IN ('Open', 'In Progress') THEN st.ticket_id END) as open_tickets
        FROM customers c
        LEFT JOIN sales s ON c.customer_id = s.customer_id
        LEFT JOIN support_tickets st ON c.customer_id = st.customer_id
        GROUP BY c.customer_id
    """
    
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    print(f"✅ Loaded {len(df)} customer records")
    return df


def prepare_features(df):
    """Prepare features for model training."""
    print("🔧 Preparing features...")
    
    # Create target variable (1 = Churned, 0 = Active/Inactive)
    df['churned'] = (df['subscription_status'] == 'Churned').astype(int)
    
    # Handle missing values
    df['days_since_last_login'] = df['days_since_last_login'].fillna(df['days_since_last_login'].median())
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['subscription_tier', 'industry', 'country']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Feature engineering
    df['avg_purchase_value'] = df['total_purchase_amount'] / (df['total_purchases'] + 1)
    df['support_ticket_rate'] = df['total_support_tickets'] / (df['account_age_days'] + 1)
    df['open_ticket_ratio'] = df['open_tickets'] / (df['total_support_tickets'] + 1)
    df['spend_per_day'] = df['monthly_spend'] / 30
    df['ltv_to_monthly_ratio'] = df['total_lifetime_value'] / (df['monthly_spend'] + 1)
    
    # Select features for training
    feature_columns = [
        'subscription_tier_encoded',
        'monthly_spend',
        'total_lifetime_value',
        'account_age_days',
        'support_tickets_count',
        'days_since_last_login',
        'total_purchases',
        'total_purchase_amount',
        'total_support_tickets',
        'open_tickets',
        'avg_purchase_value',
        'support_ticket_rate',
        'open_ticket_ratio',
        'spend_per_day',
        'ltv_to_monthly_ratio',
        'industry_encoded',
        'country_encoded'
    ]
    
    X = df[feature_columns]
    y = df['churned']
    
    print(f"✅ Prepared {len(feature_columns)} features")
    print(f"   Churn rate: {y.mean():.2%}")
    
    return X, y, feature_columns, label_encoders


def train_model(X, y, feature_columns):
    """Train Random Forest churn prediction model."""
    print("\n🤖 Training Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"churn_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("features", len(feature_columns))
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Model Performance:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        
        print("\n🔝 Top 5 Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        model_data = {
            'model': model,
            'feature_columns': feature_columns,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
        }
        joblib.dump(model_data, MODEL_OUTPUT_PATH)
        
        print(f"\n✅ Model saved to {MODEL_OUTPUT_PATH}")
        
        return model, feature_importance


def main():
    """Main training pipeline."""
    print("="*60)
    print("🚀 CHURN PREDICTION MODEL TRAINING")
    print("="*60)
    
    try:
        # Load data
        df = load_data_from_db()
        
        # Prepare features
        X, y, feature_columns, label_encoders = prepare_features(df)
        
        # Train model
        model, feature_importance = train_model(X, y, feature_columns)
        
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
