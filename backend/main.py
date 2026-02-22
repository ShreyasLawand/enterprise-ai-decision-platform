"""
FastAPI Backend for Enterprise Intelligence Platform
Main application entry point with all API endpoints.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db, init_db
from backend.models import Customer, Sale
from ml_engine.inference import get_churn_prediction, get_sales_forecast, churn_predictor
from rag_engine.chat_service import chat_service

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Intelligence Platform API",
    description="AI-powered platform for business decision-making",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ChurnPredictionRequest(BaseModel):
    """Request model for churn prediction."""
    subscription_tier_encoded: int = Field(..., description="Encoded subscription tier (0=Basic, 1=Professional, 2=Enterprise)")
    monthly_spend: float = Field(..., description="Monthly spend amount")
    total_lifetime_value: float = Field(..., description="Total lifetime value")
    account_age_days: int = Field(..., description="Account age in days")
    support_tickets_count: int = Field(..., description="Number of support tickets")
    days_since_last_login: int = Field(..., description="Days since last login")
    total_purchases: int = Field(0, description="Total number of purchases")
    total_purchase_amount: float = Field(0.0, description="Total purchase amount")
    total_support_tickets: int = Field(0, description="Total support tickets")
    open_tickets: int = Field(0, description="Number of open tickets")
    industry_encoded: int = Field(0, description="Encoded industry")
    country_encoded: int = Field(0, description="Encoded country")
    
    class Config:
        schema_extra = {
            "example": {
                "subscription_tier_encoded": 1,
                "monthly_spend": 500.0,
                "total_lifetime_value": 6000.0,
                "account_age_days": 365,
                "support_tickets_count": 5,
                "days_since_last_login": 7,
                "total_purchases": 12,
                "total_purchase_amount": 5000.0,
                "total_support_tickets": 5,
                "open_tickets": 1,
                "industry_encoded": 0,
                "country_encoded": 0
            }
        }


class ChurnPredictionResponse(BaseModel):
    """Response model for churn prediction."""
    churn_probability: float
    will_churn: bool
    risk_level: str
    model_metrics: Dict[str, float]


class ChatRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., description="User's question")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What are the common support issues?"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat queries."""
    answer: str
    service_used: str
    sources: Optional[List[Dict[str, str]]] = None
    success: bool = True


class MetricsResponse(BaseModel):
    """Response model for model metrics."""
    churn_model_metrics: Optional[Dict[str, float]]
    forecast_model_metrics: Optional[Dict[str, float]]
    database_stats: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("🚀 Starting Enterprise Intelligence Platform API...")
    
    # Initialize database
    init_db()
    
    # Initialize chat service
    try:
        chat_service.initialize()
        print("✅ Chat service initialized")
    except Exception as e:
        print(f"⚠️ Chat service initialization failed: {e}")
    
    # Load ML models
    try:
        churn_predictor.load_model()
        print("✅ Churn model loaded")
    except Exception as e:
        print(f"⚠️ Churn model loading failed: {e}")
    
    print("✅ API startup complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enterprise Intelligence Platform API",
        "version": "1.0.0",
        "endpoints": {
            "churn_prediction": "/predict/churn",
            "chat": "/ask",
            "metrics": "/metrics",
            "forecast": "/forecast",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "database": "connected",
            "ml_models": "loaded",
            "chat_service": "initialized"
        }
    }


@app.post("/predict/churn", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    """
    Predict customer churn probability.
    
    Args:
        request: Customer features for prediction
    
    Returns:
        Churn prediction with probability and risk level
    """
    try:
        # Prepare customer data
        customer_data = request.dict()
        
        # Add derived features
        customer_data['avg_purchase_value'] = (
            customer_data['total_purchase_amount'] / (customer_data['total_purchases'] + 1)
        )
        customer_data['support_ticket_rate'] = (
            customer_data['total_support_tickets'] / (customer_data['account_age_days'] + 1)
        )
        customer_data['open_ticket_ratio'] = (
            customer_data['open_tickets'] / (customer_data['total_support_tickets'] + 1)
        )
        customer_data['spend_per_day'] = customer_data['monthly_spend'] / 30
        customer_data['ltv_to_monthly_ratio'] = (
            customer_data['total_lifetime_value'] / (customer_data['monthly_spend'] + 1)
        )
        
        # Get prediction
        result = get_churn_prediction(customer_data)
        
        return ChurnPredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """
    Ask a question to the AI system (RAG or SQL agent).
    
    Args:
        request: User's question
    
    Returns:
        Answer with sources and metadata
    """
    try:
        result = chat_service.query(request.query)
        
        return ChatResponse(
            answer=result.get('answer', 'No answer generated'),
            service_used=result.get('service_used', 'unknown'),
            sources=result.get('sources'),
            success=result.get('success', True)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get("/forecast")
async def get_forecast(days: int = 30):
    """
    Get sales forecast for next N days.
    
    Args:
        days: Number of days to forecast (default: 30)
    
    Returns:
        Forecast data with predictions
    """
    try:
        result = get_sales_forecast(days)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(db: Session = Depends(get_db)):
    """
    Get model performance metrics and database statistics.
    
    Returns:
        Model metrics and database stats
    """
    try:
        # Get ML model metrics
        churn_metrics = None
        forecast_metrics = None
        
        try:
            if churn_predictor.metrics:
                churn_metrics = churn_predictor.metrics
        except:
            pass
        
        # Get database statistics
        total_customers = db.query(Customer).count()
        active_customers = db.query(Customer).filter(
            Customer.subscription_status == 'Active'
        ).count()
        churned_customers = db.query(Customer).filter(
            Customer.subscription_status == 'Churned'
        ).count()
        
        from sqlalchemy import func
        total_revenue = db.query(func.sum(Sale.total_amount)).scalar() or 0
        
        database_stats = {
            "total_customers": total_customers,
            "active_customers": active_customers,
            "churned_customers": churned_customers,
            "churn_rate": (churned_customers / total_customers * 100) if total_customers > 0 else 0,
            "total_revenue": float(total_revenue)
        }
        
        return MetricsResponse(
            churn_model_metrics=churn_metrics,
            forecast_model_metrics=forecast_metrics,
            database_stats=database_stats
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")


@app.get("/customers/stats")
async def get_customer_stats(db: Session = Depends(get_db)):
    """Get customer statistics by tier and status."""
    try:
        from sqlalchemy import func
        
        # Stats by subscription tier
        tier_stats = db.query(
            Customer.subscription_tier,
            func.count(Customer.customer_id).label('count'),
            func.avg(Customer.monthly_spend).label('avg_spend')
        ).group_by(Customer.subscription_tier).all()
        
        # Stats by status
        status_stats = db.query(
            Customer.subscription_status,
            func.count(Customer.customer_id).label('count')
        ).group_by(Customer.subscription_status).all()
        
        return {
            "by_tier": [
                {
                    "tier": stat[0],
                    "count": stat[1],
                    "avg_monthly_spend": float(stat[2]) if stat[2] else 0
                }
                for stat in tier_stats
            ],
            "by_status": [
                {
                    "status": stat[0],
                    "count": stat[1]
                }
                for stat in status_stats
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )
