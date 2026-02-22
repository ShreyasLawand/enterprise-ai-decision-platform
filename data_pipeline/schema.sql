-- Enterprise Intelligence Platform Database Schema

-- Drop tables if they exist (for clean re-initialization)
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS contracts CASCADE;
DROP TABLE IF EXISTS sales CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Customers Table
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    company VARCHAR(200),
    industry VARCHAR(100),
    country VARCHAR(100),
    subscription_tier VARCHAR(50) CHECK (subscription_tier IN ('Basic', 'Professional', 'Enterprise')),
    subscription_status VARCHAR(20) CHECK (subscription_status IN ('Active', 'Inactive', 'Churned')),
    monthly_spend DECIMAL(10, 2) DEFAULT 0.00,
    total_lifetime_value DECIMAL(12, 2) DEFAULT 0.00,
    account_age_days INTEGER DEFAULT 0,
    last_login_date DATE,
    support_tickets_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sales Table
CREATE TABLE sales (
    sale_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    product_name VARCHAR(200) NOT NULL,
    product_category VARCHAR(100),
    quantity INTEGER DEFAULT 1,
    unit_price DECIMAL(10, 2) NOT NULL,
    total_amount DECIMAL(12, 2) NOT NULL,
    discount_applied DECIMAL(5, 2) DEFAULT 0.00,
    sale_date DATE NOT NULL,
    payment_method VARCHAR(50),
    sales_rep VARCHAR(100),
    region VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Support Tickets Table
CREATE TABLE support_tickets (
    ticket_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    subject VARCHAR(300) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100),
    priority VARCHAR(20) CHECK (priority IN ('Low', 'Medium', 'High', 'Critical')),
    status VARCHAR(50) CHECK (status IN ('Open', 'In Progress', 'Resolved', 'Closed')),
    assigned_to VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_time_hours INTEGER
);

-- Contracts Table
CREATE TABLE contracts (
    contract_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    contract_title VARCHAR(300) NOT NULL,
    contract_text TEXT NOT NULL,
    contract_type VARCHAR(100),
    contract_value DECIMAL(12, 2),
    start_date DATE NOT NULL,
    end_date DATE,
    renewal_status VARCHAR(50) CHECK (renewal_status IN ('Active', 'Pending', 'Expired', 'Terminated')),
    risk_level VARCHAR(20) CHECK (risk_level IN ('Low', 'Medium', 'High')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_customers_subscription_status ON customers(subscription_status);
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_sales_customer_id ON sales(customer_id);
CREATE INDEX idx_sales_sale_date ON sales(sale_date);
CREATE INDEX idx_support_tickets_customer_id ON support_tickets(customer_id);
CREATE INDEX idx_support_tickets_status ON support_tickets(status);
CREATE INDEX idx_contracts_customer_id ON contracts(customer_id);
CREATE INDEX idx_contracts_renewal_status ON contracts(renewal_status);

-- Create a view for customer analytics
CREATE OR REPLACE VIEW customer_analytics AS
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    c.subscription_tier,
    c.subscription_status,
    c.monthly_spend,
    c.total_lifetime_value,
    c.account_age_days,
    c.support_tickets_count,
    COUNT(DISTINCT s.sale_id) as total_sales_count,
    COALESCE(SUM(s.total_amount), 0) as total_sales_amount,
    COUNT(DISTINCT st.ticket_id) as open_tickets_count
FROM customers c
LEFT JOIN sales s ON c.customer_id = s.customer_id
LEFT JOIN support_tickets st ON c.customer_id = st.customer_id AND st.status IN ('Open', 'In Progress')
GROUP BY c.customer_id;
