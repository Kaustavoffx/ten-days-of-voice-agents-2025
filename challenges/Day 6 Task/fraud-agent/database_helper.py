from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_db():
    """Connect to the database"""
    client = MongoClient(os.getenv("MONGODB_URI"))
    return client['fraud_alert_system']

# --- Function 1: get_user (Matches import in fraud_agent.py) ---
def get_user(user_id):
    """Find a user by their ID"""
    user = get_db()["users"].find_one({"user_id": user_id})
    if user: 
        user.pop('_id', None) # Remove internal MongoDB ID
    return user

# --- Function 2: get_flagged_txn ---
def get_flagged_txn(user_id):
    """Find a flagged transaction for this user"""
    txn = get_db()["transactions"].find_one({
        "user_id": user_id, 
        "status": "flagged"
    })
    if txn: 
        txn.pop('_id', None)
        # Convert timestamp to string so AI can read it
        txn['timestamp'] = str(txn['timestamp'])
    return txn

# --- Function 3: update_txn_status ---
def update_txn_status(txn_id, status):
    """Block or Approve the transaction"""
    get_db()["transactions"].update_one(
        {"transaction_id": txn_id},
        {"$set": {"status": status}}
    )
    return True