import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the MongoDB URI from environment
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not set in environment")

# Connect to MongoDB
client = MongoClient(MONGO_URI)

# Pick your database
db = client["HealthScan"]

# Example collection
master_collection = db["masterSchema"]

print("Connected to MongoDB successfully!")
