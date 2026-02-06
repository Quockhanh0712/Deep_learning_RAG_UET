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

legal_collection = os.getenv("QDRANT_LEGAL_COLLECTION", "legal_rag_vn")
user_collection = os.getenv("QDRANT_USER_COLLECTION", "user_docs_private")

print(f"Connecting to {host}...")
try:
    client = QdrantClient(url=host, api_key=api_key)
    
    # 1. Create indexes for Legal Collection
    print(f"\nProcessing Legal Collection: {legal_collection}")
    try:
        # Index 'dieu' (keyword)
        print("- Creating index for 'dieu'...")
        client.create_payload_index(
            collection_name=legal_collection,
            field_name="dieu",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD
        )
        
        # Index 'so_hieu' (keyword)
        print("- Creating index for 'so_hieu'...")
        client.create_payload_index(
            collection_name=legal_collection,
            field_name="so_hieu",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD
        )
        print("✓ Legal indexes created.")
    except Exception as e:
        print(f"Error indexing legal collection: {e}")

    # 2. Create indexes for User Collection
    print(f"\nProcessing User Collection: {user_collection}")
    try:
        # Index 'user_id' (keyword) - CRITICAL for filtering
        print("- Creating index for 'user_id'...")
        client.create_payload_index(
            collection_name=user_collection,
            field_name="user_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD
        )
        print("✓ User indexes created.")
    except Exception as e:
        print(f"Error indexing user collection: {e}")

    print("\n✓ All operations completed.")

except Exception as e:
    print(f"\nCONNECTION ERROR: {e}")
