import os
from pymongo import MongoClient
import os
import dotenv

# Load .env
dotenv.load_dotenv()

# Connect to MongoDB

# Or use Atlas:
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)

# Pick your database
db = client["HealthScan"]

# Example collection
master_collection = db["masterSchema"]

print("Connected to MongoDB successfully!")
