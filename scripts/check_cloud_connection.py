import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
load_dotenv('a:\\NCKH\\.env')

host = os.getenv("QDRANT_HOST")
api_key = os.getenv("QDRANT_API_KEY")
collections_to_check = [
    os.getenv("QDRANT_LEGAL_COLLECTION", "legal_rag_vn"),
    os.getenv("QDRANT_USER_COLLECTION", "user_docs_private")
]

print(f"Connecting to {host}...")


try:
    client = QdrantClient(
        url=host,
        api_key=api_key,
    )
    
    # Check collections
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    print("Collections found:", collection_names)
    
    for collection in collections_to_check:
        # Check specific collection
        if collection in collection_names:
            info = client.get_collection(collection)
            print(f"\nCollection {collection} info:")
            print(f"- Status: {info.status}")
            print(f"- Points count: {info.points_count}")
            
            # Print config safely
            try:
                print(f"- Vector Config: {info.config.params.vectors}")
            except:
                print(f"- Config raw: {info.config}")

            # Try a dummy search
            print("\nAttempting dummy search...")
            try:
                results = client.query_points(
                    collection_name=collection,
                    query=[0.1] * 768,  # Dummy 768-dim vector
                    limit=1,
                    using="dense",  # Assuming 'dense' is the vector name
                    with_payload=False
                )
                print(f"SUCCESS: Search returned {len(results.points)} results")
            except Exception as e:
                print(f"SEARCH FAILED: {e}")
                
        else:
            print(f"\nERROR: Collection {collection} NOT FOUND!")


except Exception as e:
    print(f"\nCONNECTION ERROR: {e}")
