import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
load_dotenv('a:\\NCKH\\.env')

host = os.getenv("QDRANT_HOST")
api_key = os.getenv("QDRANT_API_KEY")
user_collection = os.getenv("QDRANT_USER_COLLECTION", "user_docs_private")

print(f"Connecting to {host}...")
try:
    client = QdrantClient(url=host, api_key=api_key)
    
    # Check if exists
    if client.collection_exists(user_collection):
        print(f"Collection {user_collection} already exists.")
    else:
        print(f"Creating collection {user_collection}...")
        client.create_collection(
            collection_name=user_collection,
            vectors_config={
                "dense": qdrant_models.VectorParams(
                    size=768,
                    distance=qdrant_models.Distance.COSINE
                )
            }
        )
        print(f"Collection {user_collection} created successfully.")
        
except Exception as e:
    print(f"ERROR: {e}")
