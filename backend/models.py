"""
SQLAlchemy ORM models for Enterprise Intelligence Platform.
"""

from sqlalchemy import Column, Integer, String, Numeric, Date, DateTime, Text, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base


class Customer(Base):
    __tablename__ = 'customers'
    
    customer_id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20))
    company = Column(String(200))
    industry = Column(String(100))
    country = Column(String(100))
    subscription_tier = Column(String(50))
    subscription_status = Column(String(20), index=True)
    monthly_spend = Column(Numeric(10, 2), default=0.00)
    total_lifetime_value = Column(Numeric(12, 2), default=0.00)
    account_age_days = Column(Integer, default=0)
    last_login_date = Column(Date)
    support_tickets_count = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    sales = relationship("Sale", back_populates="customer", cascade="all, delete-orphan")
    support_tickets = relationship("SupportTicket", back_populates="customer", cascade="all, delete-orphan")
    contracts = relationship("Contract", back_populates="customer", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint(
            "subscription_tier IN ('Basic', 'Professional', 'Enterprise')",
            name='check_subscription_tier'
        ),
        CheckConstraint(
            "subscription_status IN ('Active', 'Inactive', 'Churned')",
            name='check_subscription_status'
        ),
    )


class Sale(Base):
    __tablename__ = 'sales'
    
    sale_id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), index=True)
    product_name = Column(String(200), nullable=False)
    product_category = Column(String(100))
    quantity = Column(Integer, default=1)
    unit_price = Column(Numeric(10, 2), nullable=False)
    total_amount = Column(Numeric(12, 2), nullable=False)
    discount_applied = Column(Numeric(5, 2), default=0.00)
    sale_date = Column(Date, nullable=False, index=True)
    payment_method = Column(String(50))
    sales_rep = Column(String(100))
    region = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    customer = relationship("Customer", back_populates="sales")


class SupportTicket(Base):
    __tablename__ = 'support_tickets'
    
    ticket_id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), index=True)
    subject = Column(String(300), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100))
    priority = Column(String(20))
    status = Column(String(50), index=True)
    assigned_to = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime)
    resolution_time_hours = Column(Integer)
    
    # Relationships
    customer = relationship("Customer", back_populates="support_tickets")
    
    __table_args__ = (
        CheckConstraint(
            "priority IN ('Low', 'Medium', 'High', 'Critical')",
            name='check_priority'
        ),
        CheckConstraint(
            "status IN ('Open', 'In Progress', 'Resolved', 'Closed')",
            name='check_status'
        ),
    )


class Contract(Base):
    __tablename__ = 'contracts'
    
    contract_id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), index=True)
    contract_title = Column(String(300), nullable=False)
    contract_text = Column(Text, nullable=False)
    contract_type = Column(String(100))
    contract_value = Column(Numeric(12, 2))
    start_date = Column(Date, nullable=False)
    end_date = Column(Date)
    renewal_status = Column(String(50), index=True)
    risk_level = Column(String(20))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    customer = relationship("Customer", back_populates="contracts")
    
    __table_args__ = (
        CheckConstraint(
            "renewal_status IN ('Active', 'Pending', 'Expired', 'Terminated')",
            name='check_renewal_status'
        ),
        CheckConstraint(
            "risk_level IN ('Low', 'Medium', 'High')",
            name='check_risk_level'
        ),
    )
