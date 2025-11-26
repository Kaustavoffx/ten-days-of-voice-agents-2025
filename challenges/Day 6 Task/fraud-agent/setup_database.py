from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

def setup_sample_data():
    """Create sample transaction data for testing"""
    try:
        # Connect
        uri = os.getenv("MONGODB_URI")
        client = MongoClient(uri)
        db = client['fraud_alert_system']
        
        # Create collections
        transactions_collection = db["transactions"]
        users_collection = db["users"]
        
        # Clear existing data
        transactions_collection.delete_many({})
        users_collection.delete_many({})
        
        # 1. Create Sample User
        sample_user = {
            "user_id": "USR001",
            "name": "Kaustav",
            "phone": "+919876543210",
            "email": "kaustav@email.com",
            "account_number": "HDFC-123456",
            "daily_limit": 50000
        }
        
        # 2. Create Suspicious Transactions
        now = datetime.now()
        sample_transactions = [
            {
                "transaction_id": "TXN001",
                "user_id": "USR001",
                "amount": 45000,
                "merchant": "Unknown Crypto Exchange",
                "location": "Moscow, Russia",
                "timestamp": now - timedelta(hours=1),
                "status": "flagged",
                "risk_reason": "Unusual location & High Value"
            },
            {
                "transaction_id": "TXN002",
                "user_id": "USR001",
                "amount": 250,
                "merchant": "Zomato",
                "location": "Kolkata, India",
                "timestamp": now - timedelta(hours=2),
                "status": "approved",
                "risk_reason": "Normal behavior"
            }
        ]
        
        # Insert data
        users_collection.insert_one(sample_user)
        transactions_collection.insert_many(sample_transactions)
        
        print("✅ Database Setup Complete!")
        print("✅ Created User: USR001 (Kaustav)")
        print("✅ Created Flagged Transaction: TXN001 ($45,000 in Russia)")
        
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        print("Check your IP Whitelist in MongoDB Atlas!")

if __name__ == "__main__":
    setup_sample_data()