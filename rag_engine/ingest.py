"""
Document Ingestion Pipeline for RAG System
Loads documents (contracts, support tickets) and stores embeddings in ChromaDB.
"""

import os
import sys
from typing import List
from datetime import datetime
from sqlalchemy import create_engine
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:admin123@localhost:5432/enterprise_db')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')

# Initialize embeddings model
print("🔧 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
print("✅ Embedding model loaded")


def load_contracts_from_db() -> List[Document]:
    """Load contract documents from database."""
    print("\n📄 Loading contracts from database...")
    
    engine = create_engine(DATABASE_URL)
    
    query = """
        SELECT 
            c.contract_id,
            c.contract_title,
            c.contract_text,
            c.contract_type,
            c.contract_value,
            c.start_date,
            c.end_date,
            c.renewal_status,
            c.risk_level,
            cust.company,
            cust.industry
        FROM contracts c
        JOIN customers cust ON c.customer_id = cust.customer_id
    """
    
    import pandas as pd
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    documents = []
    for _, row in df.iterrows():
        # Create document with metadata
        doc = Document(
            page_content=f"""
Contract Title: {row['contract_title']}
Contract Type: {row['contract_type']}
Company: {row['company']}
Industry: {row['industry']}
Contract Value: ${row['contract_value']:,.2f}
Duration: {row['start_date']} to {row['end_date']}
Renewal Status: {row['renewal_status']}
Risk Level: {row['risk_level']}

Contract Details:
{row['contract_text']}
            """.strip(),
            metadata={
                'contract_id': str(row['contract_id']),
                'contract_title': row['contract_title'],
                'contract_type': row['contract_type'],
                'company': row['company'],
                'industry': row['industry'],
                'contract_value': float(row['contract_value']),
                'renewal_status': row['renewal_status'],
                'risk_level': row['risk_level'],
                'source': 'contracts'
            }
        )
        documents.append(doc)
    
    print(f"✅ Loaded {len(documents)} contracts")
    return documents


def load_support_tickets_from_db() -> List[Document]:
    """Load support ticket documents from database."""
    print("\n🎫 Loading support tickets from database...")
    
    engine = create_engine(DATABASE_URL)
    
    query = """
        SELECT 
            st.ticket_id,
            st.subject,
            st.description,
            st.category,
            st.priority,
            st.status,
            st.created_at,
            st.resolved_at,
            st.resolution_time_hours,
            cust.company,
            cust.industry,
            cust.subscription_tier
        FROM support_tickets st
        JOIN customers cust ON st.customer_id = cust.customer_id
        LIMIT 500
    """
    
    import pandas as pd
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    documents = []
    for _, row in df.iterrows():
        # Create document with metadata
        doc = Document(
            page_content=f"""
Ticket Subject: {row['subject']}
Category: {row['category']}
Priority: {row['priority']}
Status: {row['status']}
Company: {row['company']}
Industry: {row['industry']}
Subscription Tier: {row['subscription_tier']}

Description:
{row['description']}
            """.strip(),
            metadata={
                'ticket_id': str(row['ticket_id']),
                'subject': row['subject'],
                'category': row['category'],
                'priority': row['priority'],
                'status': row['status'],
                'company': row['company'],
                'industry': row['industry'],
                'source': 'support_tickets'
            }
        )
        documents.append(doc)
    
    print(f"✅ Loaded {len(documents)} support tickets")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better retrieval."""
    print("\n✂️ Chunking documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def create_vector_store(documents: List[Document]):
    """Create and persist ChromaDB vector store."""
    print("\n🗄️ Creating vector store...")
    
    # Create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="enterprise_documents"
    )
    
    print(f"✅ Vector store created with {len(documents)} documents")
    print(f"   Persisted to: {CHROMA_PERSIST_DIR}")
    
    return vectorstore


def main():
    """Main ingestion pipeline."""
    print("="*60)
    print("🚀 DOCUMENT INGESTION PIPELINE")
    print("="*60)
    
    try:
        # Load documents from database
        contracts = load_contracts_from_db()
        support_tickets = load_support_tickets_from_db()
        
        # Combine all documents
        all_documents = contracts + support_tickets
        print(f"\n📚 Total documents loaded: {len(all_documents)}")
        
        # Chunk documents
        chunks = chunk_documents(all_documents)
        
        # Create vector store
        vectorstore = create_vector_store(chunks)
        
        # Test retrieval
        print("\n🔍 Testing retrieval...")
        test_query = "What are the common support issues?"
        results = vectorstore.similarity_search(test_query, k=3)
        
        print(f"   Query: '{test_query}'")
        print(f"   Retrieved {len(results)} relevant documents")
        
        print("\n" + "="*60)
        print("✅ INGESTION COMPLETE!")
        print("="*60)
        print(f"Total documents indexed: {len(chunks)}")
        print(f"Vector store location: {CHROMA_PERSIST_DIR}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
