"""
Chat Service for RAG and SQL Agent
Provides unified interface for querying documents and database using LOCAL OLLAMA.
"""

import os
from typing import Dict, Any
from sqlalchemy import create_engine, text

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

from dotenv import load_dotenv

load_dotenv()

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:admin123@localhost:5432/enterprise_db')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')

# Define the Ollama model you pulled
OLLAMA_MODEL = "llama3.2:1b"

class RAGChatService:
    """RAG-based chat service for document queries."""
    
    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
    
    def initialize(self):
        """Initialize the RAG components."""
        print(f"🔧 Initializing RAG service with Local Ollama ({OLLAMA_MODEL})...")
        
        # Load embeddings (Keeping HuggingFace for fast, local vector math)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="enterprise_documents"
        )
        
        # Initialize Local Ollama LLM
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.7,
        )
        print("✅ Local Ollama LLM initialized for RAG")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        if self.vectorstore is None:
            self.initialize()
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=3)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        answer = ""
        if self.llm:
            prompt = f"""You are an enterprise AI assistant. Use the following context to answer the user's question. If you don't know the answer, say "I don't know".
            
            Context:
            {context_text}
            
            Question: {question}
            
            Answer:"""
            
            try:
                # Generate Answer using local hardware
                response = self.llm.invoke(prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error communicating with local Ollama: {str(e)}\nMake sure Ollama app is running in the background!"
        else:
            answer = "LLM not configured."
        
        # Extract sources
        source_info = []
        for doc in docs:
            source_info.append({
                'source': doc.metadata.get('source', 'unknown'),
                'title': doc.metadata.get('contract_title') or doc.metadata.get('subject', 'N/A'),
                'company': doc.metadata.get('company', 'N/A')
            })
        
        return {
            'answer': answer.strip(),
            'sources': source_info,
            'service_used': 'rag (Local Ollama)'
        }


class SQLAgentService:
    """SQL Agent for database queries."""
    
    def __init__(self):
        self.db = None
        self.agent = None
        self.llm = None
    
    def initialize(self):
        """Initialize the SQL agent."""
        print(f"🔧 Initializing SQL agent with Local Ollama ({OLLAMA_MODEL})...")
        self.db = SQLDatabase.from_uri(DATABASE_URL)
        
        # Temperature 0 is crucial for SQL so it doesn't "hallucinate" fake syntax
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0, 
        )
        
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            agent_type="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True
        )
        print("✅ Local SQL agent initialized")
    
    def query(self, question: str) -> Dict[str, Any]:
        if self.db is None:
            self.initialize()
        
        try:
            if self.agent:
                result = self.agent.run(question)
                return {'answer': result, 'success': True}
            else:
                return {'answer': "SQL Agent not configured", 'success': False}
        except Exception as e:
            # Fallback for SQL queries if local Llama 3 struggles with LangChain's parsing
            return self._fallback_query(question)
            
    def _fallback_query(self, question: str) -> Dict[str, Any]:
        """Simple fallback if the LLM SQL agent fails to parse."""
        question_lower = question.lower()
        engine = create_engine(DATABASE_URL)
        answer = "I couldn't safely generate a SQL query for that. Try asking about 'revenue', 'churn', or 'total customers'."
        
        try:
            if 'revenue' in question_lower or 'sales' in question_lower:
                # FIXED: Changed 'amount' to 'total_amount' to match your models.py schema
                query = text("SELECT SUM(total_amount) FROM sales")
                with engine.connect() as conn:
                    result = conn.execute(query).fetchone()
                    if result and result[0]:
                        answer = f"Total historical revenue is ${result[0]:,.2f}"
            elif 'customer' in question_lower and 'count' in question_lower:
                query = text("SELECT COUNT(*) FROM customers")
                with engine.connect() as conn:
                    result = conn.execute(query).fetchone()
                    if result and result[0]:
                        answer = f"There are {result[0]:,} total customers."
            elif 'churn' in question_lower:
                query = text("SELECT COUNT(*) FROM customers WHERE subscription_status = 'Churned'")
                with engine.connect() as conn:
                    result = conn.execute(query).fetchone()
                    if result and result[0]:
                        answer = f"There are {result[0]:,} churned customers."
        except Exception as e:
            # Added error handling so it doesn't 500 crash if the DB fails
            print(f"Database error in fallback: {str(e)}")
            answer = "I encountered a database error while looking that up."
        finally:
            engine.dispose()
            
        return {'answer': answer, 'success': True}

class UnifiedChatService:
    def __init__(self):
        self.rag_service = RAGChatService()
        self.sql_service = SQLAgentService()
    
    def initialize(self):
        self.rag_service.initialize()
        self.sql_service.initialize()
    
    def query(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()
        sql_keywords = ['revenue', 'sales', 'total', 'count', 'average', 'how many']
        
        if any(k in question_lower for k in sql_keywords):
            result = self.sql_service.query(question)
            result['service_used'] = 'sql_agent (Local)'
        else:
            result = self.rag_service.query(question)
            result['service_used'] = 'rag (Local)'
        return result

chat_service = UnifiedChatService()