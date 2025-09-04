from pymongo import MongoClient

# Use local MongoDB
MONGO_URI = "mongodb://localhost:27017"

# Or use Atlas:
MONGO_URI = "mongodb+srv://akshay:Akshay1234@cluster0.wwuryc3.mongodb.net/HealthScan"

client = MongoClient(MONGO_URI)

# Pick your database
db = client["HealthScan"]

# Example collections
master_collection = db["masterSchema"]
