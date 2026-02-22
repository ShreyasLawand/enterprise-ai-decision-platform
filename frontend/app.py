"""
Streamlit Frontend for Enterprise Intelligence Platform
Interactive dashboard with visualizations, AI chat, and predictions.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Enterprise Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful, modern UI
st.markdown("""
    <style>
    /* Main Title Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1f77b4, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding-top: 1rem;
    }
    
    /* Sidebar Branding */
    .sidebar-brand {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(128,128,128,0.2);
        margin-bottom: 1rem;
    }
    
    /* Metric Cards Enhancement */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 1.2rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


def get_api_data(endpoint: str):
    """Fetch data from API endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def post_api_data(endpoint: str, data: dict):
    """Post data to API endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


# Sidebar
with st.sidebar:
    # Replaced the deprecated st.image with a clean HTML header
    st.markdown('<div class="sidebar-brand">🧠 Enterprise AI</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌟 Platform Features")
    st.markdown("""
    - 📊 **Real-time Analytics**
    - 🤖 **AI-Powered Predictions**
    - 💬 **Intelligent Chat**
    - 📈 **Sales Forecasting**
    """)
    st.markdown("---")
    
    # Health check
    health = get_api_data("/health")
    if health and health.get("status") == "healthy":
        st.success("✅ System Online")
    else:
        st.error("❌ System Offline (Check Backend)")


# Main header
st.markdown('<div class="main-header">Enterprise Intelligence Platform</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 AI Chat", "🎯 Predictions"])


# TAB 1: Dashboard
with tab1:
    
    # Fetch metrics
    metrics_data = get_api_data("/metrics")
    customer_stats = get_api_data("/customers/stats")
    
    if metrics_data:
        db_stats = metrics_data.get("database_stats", {})
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Customers",
                value=f"{db_stats.get('total_customers', 0):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Active Customers",
                value=f"{db_stats.get('active_customers', 0):,}",
                delta=None
            )
        
        with col3:
            churn_rate = db_stats.get('churn_rate', 0)
            st.metric(
                label="Churn Rate",
                value=f"{churn_rate:.2f}%",
                delta=f"{churn_rate:.2f}%",
                delta_color="inverse"
            )
        
        with col4:
            revenue = db_stats.get('total_revenue', 0)
            st.metric(
                label="Total Revenue",
                value=f"${revenue:,.0f}",
                delta=None
            )
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Customers by Subscription Tier")
            if customer_stats and 'by_tier' in customer_stats:
                tier_df = pd.DataFrame(customer_stats['by_tier'])
                fig = px.pie(
                    tier_df,
                    values='count',
                    names='tier',
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Customers by Status")
            if customer_stats and 'by_status' in customer_stats:
                status_df = pd.DataFrame(customer_stats['by_status'])
                fig = px.bar(
                    status_df,
                    x='status',
                    y='count',
                    color='status',
                    color_discrete_map={
                        'Active': '#2ecc71',
                        'Inactive': '#f39c12',
                        'Churned': '#e74c3c'
                    }
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
        
        # Sales forecast
        st.markdown("---")
        st.subheader("📈 Sales Forecast (Next 30 Days)")
        
        forecast_data = get_api_data("/forecast?days=30")
        if forecast_data:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'forecast_data' in forecast_data:
                    forecast_df = pd.DataFrame(forecast_data['forecast_data'])
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=forecast_df['ds'],
                        y=forecast_df['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_df['ds'],
                        y=forecast_df['yhat_upper'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_df['ds'],
                        y=forecast_df['yhat_lower'],
                        mode='lines',
                        name='Lower Bound',
                        line=dict(width=0),
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        fill='tonexty',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Revenue ($)',
                        hovermode='x unified',
                        margin=dict(t=20, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.info("### Forecast Summary")
                st.metric(
                    "Total Expected",
                    f"${forecast_data.get('total_forecast', 0):,.2f}"
                )
                st.metric(
                    "Daily Average",
                    f"${forecast_data.get('daily_average', 0):,.2f}"
                )
                
                if 'model_metrics' in forecast_data:
                    st.markdown("---")
                    st.markdown("**Model Confidence (Prophet)**")
                    metrics = forecast_data['model_metrics']
                    st.caption(f"R² Score: {metrics.get('r2_score', 0):.4f}")
                    st.caption(f"MAPE Error: {metrics.get('mape', 0):.2f}%")


# TAB 2: AI Chat
with tab2:
    st.subheader("💬 AI-Powered Business Intelligence")
    st.caption("Ask natural language questions about your data, contracts, or support tickets.")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    # Example questions buttons (only show if no messages yet)
    if not st.session_state.messages:
        st.markdown("#### 💡 Example Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 What is the total revenue?", use_container_width=True):
                st.session_state.example_prompt = "What is the total revenue?"
            if st.button("📄 Summarize contract risks", use_container_width=True):
                st.session_state.example_prompt = "Summarize the risks in our contracts"
        
        with col2:
            if st.button("🎫 Common support issues?", use_container_width=True):
                st.session_state.example_prompt = "What are the most common support issues?"
            if st.button("👥 How many customers churned?", use_container_width=True):
                st.session_state.example_prompt = "How many customers have churned?"
                
    # Determine the prompt (either from chat_input or a clicked example button)
    user_prompt = st.chat_input("Ask a question about your business data...")
    if 'example_prompt' in st.session_state:
        user_prompt = st.session_state.example_prompt
        del st.session_state.example_prompt  # clear it so it only runs once
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"- **{source.get('source')}**: {source.get('title')} ({source.get('company')})")

    # React to user input
    if user_prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching database and documents..."):
                response = post_api_data("/ask", {"query": user_prompt})
                
                if response:
                    answer = response.get('answer', 'No response generated.')
                    service = response.get('service_used', 'unknown')
                    
                    # Format the response beautifully
                    formatted_answer = f"**Agent Used:** `{service}`\n\n{answer}"
                    st.markdown(formatted_answer)
                    
                    sources = response.get('sources')
                    if sources:
                        with st.expander("📚 View Document Sources"):
                            for source in sources:
                                st.write(f"- **{source.get('source')}**: {source.get('title')} ({source.get('company')})")
                                
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_answer,
                        "sources": sources
                    })


# TAB 3: Predictions
with tab3:
    st.header("🎯 Customer Churn Prediction")
    st.markdown("Enter customer data to run the Random Forest ML pipeline and predict churn risk.")
    
    with st.form("churn_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Customer Information")
            subscription_tier = st.selectbox("Subscription Tier", ["Basic", "Professional", "Enterprise"], index=1)
            monthly_spend = st.number_input("Monthly Spend ($)", min_value=0.0, value=500.0, step=50.0)
            account_age_days = st.number_input("Account Age (days)", min_value=0, value=365, step=30)
            days_since_last_login = st.number_input("Days Since Last Login", min_value=0, value=7, step=1)
        
        with col2:
            st.subheader("📈 Usage & Support")
            total_purchases = st.number_input("Total Purchases", min_value=0, value=12, step=1)
            total_purchase_amount = st.number_input("Total Purchase Amount ($)", min_value=0.0, value=5000.0, step=100.0)
            support_tickets_count = st.number_input("Total Support Tickets", min_value=0, value=5, step=1)
            open_tickets = st.number_input("Open Tickets", min_value=0, value=1, step=1)
            
        submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary", use_container_width=True)
    
    if submitted:
        tier_encoding = {"Basic": 0, "Professional": 1, "Enterprise": 2}
        
        request_data = {
            "subscription_tier_encoded": tier_encoding[subscription_tier],
            "monthly_spend": monthly_spend,
            "total_lifetime_value": monthly_spend * (account_age_days / 30),
            "account_age_days": account_age_days,
            "support_tickets_count": support_tickets_count,
            "days_since_last_login": days_since_last_login,
            "total_purchases": total_purchases,
            "total_purchase_amount": total_purchase_amount,
            "total_support_tickets": support_tickets_count,
            "open_tickets": open_tickets,
            "industry_encoded": 0,
            "country_encoded": 0
        }
        
        with st.spinner("🔮 Processing through ML Model..."):
            prediction = post_api_data("/predict/churn", request_data)
        
        if prediction:
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                churn_prob = prediction.get('churn_probability', 0)
                st.metric("Churn Probability", f"{churn_prob * 100:.1f}%")
            
            with res_col2:
                risk_level = prediction.get('risk_level', 'Unknown')
                risk_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                st.metric("Risk Level", f"{risk_colors.get(risk_level, '⚪')} {risk_level}")
            
            with res_col3:
                will_churn = prediction.get('will_churn', False)
                st.metric("Model Classification", "Will Churn" if will_churn else "Will Stay")
            
            # Visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Gauge"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgray"},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ecc71"},
                        {'range': [30, 60], 'color': "#f1c40f"},
                        {'range': [60, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': churn_prob * 100
                    }
                }
            ))
            fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Automated Action Plan")
            if risk_level == "High":
                st.error("""
                **🚨 Immediate Action Required:**
                - Schedule a personal check-in call with the account manager.
                - Offer special retention incentives or discounts.
                - Prioritize and resolve any open support tickets immediately.
                """)
            elif risk_level == "Medium":
                st.warning("""
                **⚠️ Monitor Closely:**
                - Send automated engagement emails with product tips.
                - Offer targeted training or onboarding support.
                - Check in on satisfaction levels via survey.
                """)
            else:
                st.success("""
                **✅ Maintain Engagement:**
                - Continue regular cadence of communication.
                - Customer is primed for upsell opportunities.
                - Request a testimonial or referral.
                """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9rem;'>"
    "Built for the Enterprise AI Decision Platform | Powered by FastAPI, PostgreSQL, and LLMs"
    "</div>",
    unsafe_allow_html=True
)