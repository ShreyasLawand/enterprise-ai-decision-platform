# 🧠 Enterprise Intelligence Platform

A production-grade AI-powered platform that combines SQL data warehousing, Machine Learning pipelines, and Generative AI (RAG) to automate business decision-making.

## 🌟 Features

### Data Engineering Layer
- **PostgreSQL Database** with comprehensive schema for customers, sales, support tickets, and contracts
- **Synthetic Data Generation** with 500+ customers and 2000+ transactions using Faker
- **Automated Seeding** for instant deployment

### Machine Learning Layer
- **Churn Prediction** using Random Forest classifier with 17 engineered features
- **Sales Forecasting** using Prophet for time-series prediction
- **MLflow Integration** for experiment tracking and model versioning
- **Model Inference API** for real-time predictions

### GenAI & RAG Layer
- **Document Ingestion** pipeline for contracts and support tickets
- **ChromaDB Vector Store** for semantic search
- **RAG QA Chain** using LangChain for document queries
- **SQL Agent** for natural language database queries
- **Intelligent Routing** between RAG and SQL based on query type

### API Layer (FastAPI)
- **POST /predict/churn** - Customer churn probability prediction
- **POST /ask** - Natural language queries (RAG + SQL)
- **GET /forecast** - Sales forecasting for next N days
- **GET /metrics** - Model performance and database statistics
- **GET /customers/stats** - Customer analytics by tier and status

### Frontend (Streamlit)
- **📊 Dashboard Tab** - Real-time analytics with interactive charts
- **💬 AI Chat Tab** - Conversational interface for data queries
- **🎯 Predictions Tab** - Interactive churn risk assessment

## 🏗️ Architecture

```
enterprise-ai-platform/
├── docker-compose.yml          # PostgreSQL & MLflow services
├── .env                        # Configuration & API keys
├── requirements.txt            # Python dependencies
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── database.py             # Database connection
│   ├── models.py               # SQLAlchemy ORM models
│   └── api/                    # API routers
├── data_pipeline/
│   ├── schema.sql              # Database schema
│   └── seed_data.py            # Synthetic data generator
├── ml_engine/
│   ├── train_churn.py          # Churn prediction training
│   ├── train_forecast.py       # Sales forecasting training
│   └── inference.py            # Model inference
├── rag_engine/
│   ├── ingest.py               # Document ingestion
│   └── chat_service.py         # RAG & SQL agent
└── frontend/
    └── app.py                  # Streamlit dashboard
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- 8GB RAM minimum
- OpenAI API key (optional, fallback mode available)

### Step 1: Clone and Setup

```bash
cd "Enterprise Intelligence Platform"

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

Edit `.env` file and add your OpenAI API key (optional):
```bash
OPENAI_API_KEY=your-api-key-here
```

### Step 3: Start Infrastructure

```bash
# Start PostgreSQL (and optional MLflow)
docker compose up -d

# Wait for services to be ready (30 seconds)
# Windows PowerShell:
Start-Sleep -Seconds 30
```

### Step 4: Initialize Database

```bash
# Seed the database with synthetic data
python data_pipeline/seed_data.py
```

### Step 5: Train ML Models

```bash
# Train churn prediction model
python ml_engine/train_churn.py

# Train sales forecasting model
python ml_engine/train_forecast.py
```

### Step 6: Ingest Documents for RAG

```bash
# Ingest contracts and support tickets into ChromaDB
python rag_engine/ingest.py
```

### Step 7: Start Backend API

```bash
# Start FastAPI server
python backend/main.py
```

The API will be available at `http://localhost:8000`

### Step 8: Launch Frontend

```bash
# In a new terminal, start Streamlit
streamlit run frontend/app.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Using the Platform

### Dashboard Tab
- View real-time business metrics
- Analyze customer distribution by tier and status
- Review 30-day sales forecast with confidence intervals
- Monitor churn rates and revenue

### AI Chat Tab
Ask questions like:
- "What is the total revenue for last month?"
- "Summarize the risks in our contracts"
- "What are the most common support issues?"
- "How many customers have churned?"

### Predictions Tab
Enter customer data to get:
- Churn probability percentage
- Risk level (Low/Medium/High)
- Actionable recommendations

## 🔧 API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📈 MLflow Tracking

View experiment tracking and model metrics:
- **MLflow UI**: http://localhost:5000

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Python 3.9+ |
| Database | PostgreSQL 15 |
| Vector DB | ChromaDB |
| ML & Data | Pandas, Scikit-Learn, Prophet |
| ML Tracking | MLflow |
| GenAI | LangChain, OpenAI API |
| Embeddings | SentenceTransformers |
| Frontend | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |

## 📝 Model Performance

### Churn Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: 17 engineered features
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Training Data**: 500 customers with historical data

### Sales Forecasting Model
- **Algorithm**: Prophet (Facebook)
- **Features**: Daily revenue, transaction count, average value
- **Metrics**: MAE, RMSE, R², MAPE
- **Forecast Horizon**: 30 days

## 🔐 Security Notes

- Change default database credentials in `.env`
- Never commit `.env` file to version control
- Use environment-specific API keys
- Enable authentication for production deployment

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker ps

# View logs
docker-compose logs postgres
```

### Model Not Found Errors
```bash
# Ensure models are trained
ls ml_engine/models/

# Retrain if needed
python ml_engine/train_churn.py
```

### ChromaDB Issues
```bash
# Reingest documents
rm -rf chroma_db/
python rag_engine/ingest.py
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## 🤝 Contributing

This is a demonstration platform. For production use:
1. Implement proper authentication
2. Add comprehensive error handling
3. Set up monitoring and logging
4. Configure backup strategies
5. Implement CI/CD pipelines

## 📄 License

This project is for educational and demonstration purposes.

---

**Built with ❤️ using AI & Machine Learning**
