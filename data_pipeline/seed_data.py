"""
Seed Data Generator for Enterprise Intelligence Platform
Generates realistic synthetic data for customers, sales, support tickets, and contracts.
"""

import os
import random
from datetime import datetime, timedelta
from faker import Faker
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Faker
fake = Faker()

# Database connection parameters
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'enterprise_db'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
}

# Configuration
NUM_CUSTOMERS = 500
NUM_SALES = 2000
NUM_SUPPORT_TICKETS = 800
NUM_CONTRACTS = 400

# Data generation helpers
SUBSCRIPTION_TIERS = ['Basic', 'Professional', 'Enterprise']
SUBSCRIPTION_STATUSES = ['Active', 'Inactive', 'Churned']
INDUSTRIES = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Education', 'Consulting']
PRODUCT_CATEGORIES = ['Software License', 'Consulting Services', 'Support Package', 'Training', 'Hardware']
PRODUCTS = {
    'Software License': ['Enterprise Suite', 'Pro Edition', 'Starter Pack', 'Analytics Module'],
    'Consulting Services': ['Strategy Consulting', 'Implementation Services', 'Custom Development'],
    'Support Package': ['24/7 Premium Support', 'Business Hours Support', 'Email Support'],
    'Training': ['Onboarding Training', 'Advanced Workshop', 'Certification Program'],
    'Hardware': ['Server Appliance', 'IoT Devices', 'Security Hardware']
}
PAYMENT_METHODS = ['Credit Card', 'Wire Transfer', 'ACH', 'PayPal', 'Invoice']
TICKET_CATEGORIES = ['Technical Issue', 'Billing Question', 'Feature Request', 'Account Management', 'Bug Report']
TICKET_PRIORITIES = ['Low', 'Medium', 'High', 'Critical']
TICKET_STATUSES = ['Open', 'In Progress', 'Resolved', 'Closed']
CONTRACT_TYPES = ['Annual Subscription', 'Multi-Year Agreement', 'Professional Services', 'Partnership Agreement']
RENEWAL_STATUSES = ['Active', 'Pending', 'Expired', 'Terminated']
RISK_LEVELS = ['Low', 'Medium', 'High']


def generate_customers(num_customers):
    """Generate synthetic customer data."""
    customers = []
    for _ in range(num_customers):
        subscription_tier = random.choice(SUBSCRIPTION_TIERS)
        subscription_status = random.choices(
            SUBSCRIPTION_STATUSES, 
            weights=[0.7, 0.2, 0.1]  # 70% active, 20% inactive, 10% churned
        )[0]
        
        # Calculate monthly spend based on tier
        tier_spend = {'Basic': (50, 200), 'Professional': (200, 800), 'Enterprise': (800, 5000)}
        monthly_spend = round(random.uniform(*tier_spend[subscription_tier]), 2)
        
        # Account age in days (0 to 1095 days = 3 years)
        account_age_days = random.randint(0, 1095)
        
        # Calculate lifetime value
        total_lifetime_value = round(monthly_spend * (account_age_days / 30), 2)
        
        # Last login date
        last_login_date = fake.date_between(start_date='-90d', end_date='today')
        
        # Support tickets count
        support_tickets_count = random.randint(0, 20)
        
        customer = (
            fake.first_name(),
            fake.last_name(),
            fake.unique.email(),
            fake.phone_number()[:20],
            fake.company(),
            random.choice(INDUSTRIES),
            fake.country(),
            subscription_tier,
            subscription_status,
            monthly_spend,
            total_lifetime_value,
            account_age_days,
            last_login_date,
            support_tickets_count
        )
        customers.append(customer)
    
    return customers


def generate_sales(num_sales, customer_ids):
    """Generate synthetic sales transactions."""
    sales = []
    start_date = datetime.now() - timedelta(days=730)  # 2 years of data
    
    for _ in range(num_sales):
        customer_id = random.choice(customer_ids)
        category = random.choice(PRODUCT_CATEGORIES)
        product_name = random.choice(PRODUCTS[category])
        quantity = random.randint(1, 10)
        
        # Price ranges by category
        price_ranges = {
            'Software License': (500, 5000),
            'Consulting Services': (1000, 10000),
            'Support Package': (200, 2000),
            'Training': (300, 3000),
            'Hardware': (500, 8000)
        }
        unit_price = round(random.uniform(*price_ranges[category]), 2)
        total_amount = round(unit_price * quantity, 2)
        discount_applied = round(random.uniform(0, 15), 2)
        
        # Apply discount
        total_amount = round(total_amount * (1 - discount_applied / 100), 2)
        
        sale_date = fake.date_between(start_date=start_date, end_date='today')
        
        sale = (
            customer_id,
            product_name,
            category,
            quantity,
            unit_price,
            total_amount,
            discount_applied,
            sale_date,
            random.choice(PAYMENT_METHODS),
            fake.name(),
            fake.state()
        )
        sales.append(sale)
    
    return sales


def generate_support_tickets(num_tickets, customer_ids):
    """Generate synthetic support tickets."""
    tickets = []
    
    ticket_subjects = [
        "Unable to login to account",
        "Billing discrepancy on invoice",
        "Feature request: Export to CSV",
        "Application crashes on startup",
        "Need help with API integration",
        "Password reset not working",
        "Performance issues with dashboard",
        "Request for additional user licenses",
        "Data synchronization error",
        "Mobile app not loading"
    ]
    
    for _ in range(num_tickets):
        customer_id = random.choice(customer_ids)
        subject = random.choice(ticket_subjects)
        description = fake.paragraph(nb_sentences=5)
        category = random.choice(TICKET_CATEGORIES)
        priority = random.choices(
            TICKET_PRIORITIES,
            weights=[0.4, 0.35, 0.2, 0.05]  # Most tickets are low/medium priority
        )[0]
        status = random.choices(
            TICKET_STATUSES,
            weights=[0.15, 0.25, 0.35, 0.25]  # Distribution across statuses
        )[0]
        
        created_at = fake.date_time_between(start_date='-180d', end_date='now')
        
        # If resolved or closed, set resolution time
        resolved_at = None
        resolution_time_hours = None
        if status in ['Resolved', 'Closed']:
            resolution_time_hours = random.randint(1, 72)
            resolved_at = created_at + timedelta(hours=resolution_time_hours)
        
        ticket = (
            customer_id,
            subject,
            description,
            category,
            priority,
            status,
            fake.name(),
            created_at,
            resolved_at,
            resolution_time_hours
        )
        tickets.append(ticket)
    
    return tickets


def generate_contracts(num_contracts, customer_ids):
    """Generate synthetic contract documents."""
    contracts = []
    
    contract_templates = [
        "This Software License Agreement grants {company} the right to use our enterprise software platform for a period of {duration} months. The total contract value is ${value}. Key terms include: unlimited user access, 24/7 support, quarterly business reviews, and data security compliance. Renewal terms require 60-day notice. Termination clauses include breach of payment terms or violation of usage policies.",
        "Professional Services Agreement between our company and {company} for implementation and consulting services. Contract duration: {duration} months. Total value: ${value}. Deliverables include system integration, custom development, and training. Payment terms: 30% upfront, 40% at milestone completion, 30% upon final delivery. Risk factors: scope creep, resource availability, and timeline dependencies.",
        "Annual Subscription Agreement for {company} covering software access and support services. Term: {duration} months. Annual value: ${value}. Includes automatic renewal unless cancelled with 30-day notice. Service level agreement guarantees 99.9% uptime. Pricing adjustments may occur annually based on CPI. Early termination fees apply.",
        "Partnership Agreement establishing a strategic relationship with {company}. Duration: {duration} months. Total commitment: ${value}. Terms include revenue sharing, co-marketing initiatives, and joint product development. Confidentiality and non-compete clauses are in effect. Either party may terminate with 90-day written notice."
    ]
    
    for _ in range(num_contracts):
        customer_id = random.choice(customer_ids)
        duration = random.choice([12, 24, 36])
        contract_value = round(random.uniform(10000, 500000), 2)
        
        contract_text = random.choice(contract_templates).format(
            company=fake.company(),
            duration=duration,
            value=f"{contract_value:,.2f}"
        )
        
        start_date = fake.date_between(start_date='-730d', end_date='today')
        end_date = start_date + timedelta(days=duration * 30)
        
        # Determine renewal status based on end date
        if end_date < datetime.now().date():
            renewal_status = random.choice(['Expired', 'Terminated'])
        elif end_date < (datetime.now() + timedelta(days=60)).date():
            renewal_status = 'Pending'
        else:
            renewal_status = 'Active'
        
        risk_level = random.choices(
            RISK_LEVELS,
            weights=[0.6, 0.3, 0.1]  # Most contracts are low risk
        )[0]
        
        contract = (
            customer_id,
            f"{random.choice(CONTRACT_TYPES)} - {fake.catch_phrase()}",
            contract_text,
            random.choice(CONTRACT_TYPES),
            contract_value,
            start_date,
            end_date,
            renewal_status,
            risk_level
        )
        contracts.append(contract)
    
    return contracts


def seed_database():
    """Main function to seed the database with synthetic data."""
    print("🚀 Starting database seeding process...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✅ Connected to database")
        
        # Generate and insert customers
        print(f"\n📊 Generating {NUM_CUSTOMERS} customers...")
        customers = generate_customers(NUM_CUSTOMERS)
        
        customer_insert_query = """
            INSERT INTO customers (
                first_name, last_name, email, phone, company, industry, country,
                subscription_tier, subscription_status, monthly_spend, total_lifetime_value,
                account_age_days, last_login_date, support_tickets_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_batch(cursor, customer_insert_query, customers)
        conn.commit()
        print(f"✅ Inserted {NUM_CUSTOMERS} customers")
        
        # Get customer IDs
        cursor.execute("SELECT customer_id FROM customers")
        customer_ids = [row[0] for row in cursor.fetchall()]
        
        # Generate and insert sales
        print(f"\n💰 Generating {NUM_SALES} sales transactions...")
        sales = generate_sales(NUM_SALES, customer_ids)
        
        sales_insert_query = """
            INSERT INTO sales (
                customer_id, product_name, product_category, quantity, unit_price,
                total_amount, discount_applied, sale_date, payment_method, sales_rep, region
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_batch(cursor, sales_insert_query, sales)
        conn.commit()
        print(f"✅ Inserted {NUM_SALES} sales transactions")
        
        # Generate and insert support tickets
        print(f"\n🎫 Generating {NUM_SUPPORT_TICKETS} support tickets...")
        tickets = generate_support_tickets(NUM_SUPPORT_TICKETS, customer_ids)
        
        tickets_insert_query = """
            INSERT INTO support_tickets (
                customer_id, subject, description, category, priority, status,
                assigned_to, created_at, resolved_at, resolution_time_hours
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_batch(cursor, tickets_insert_query, tickets)
        conn.commit()
        print(f"✅ Inserted {NUM_SUPPORT_TICKETS} support tickets")
        
        # Generate and insert contracts
        print(f"\n📄 Generating {NUM_CONTRACTS} contracts...")
        contracts = generate_contracts(NUM_CONTRACTS, customer_ids)
        
        contracts_insert_query = """
            INSERT INTO contracts (
                customer_id, contract_title, contract_text, contract_type, contract_value,
                start_date, end_date, renewal_status, risk_level
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_batch(cursor, contracts_insert_query, contracts)
        conn.commit()
        print(f"✅ Inserted {NUM_CONTRACTS} contracts")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("📈 DATABASE SEEDING COMPLETE!")
        print("="*60)
        
        cursor.execute("SELECT COUNT(*) FROM customers")
        print(f"Total Customers: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM sales")
        print(f"Total Sales: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM support_tickets")
        print(f"Total Support Tickets: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM contracts")
        print(f"Total Contracts: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT SUM(total_amount) FROM sales")
        total_revenue = cursor.fetchone()[0]
        print(f"Total Revenue: ${total_revenue:,.2f}")
        
        print("="*60)
        
        cursor.close()
        conn.close()
        print("\n✅ Database connection closed")
        
    except Exception as e:
        print(f"\n❌ Error seeding database: {e}")
        raise


if __name__ == "__main__":
    seed_database()
