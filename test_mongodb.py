import os
from dotenv import load_dotenv
from pymongo import MongoClient

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Fetch the MongoDB URI securely
mongo_uri = os.getenv("MONGO_URI")

# Optional but recommended: Add a quick check to ensure it loaded properly
if not mongo_uri:
    raise ValueError("MONGO_URI environment variable is missing. Check your .env file.")

# 3. Connect to the database
client = MongoClient(mongo_uri)
db = client["NAVAI"]

print("Connected to MongoDB successfully!")