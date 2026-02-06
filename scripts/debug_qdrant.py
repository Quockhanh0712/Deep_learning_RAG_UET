import sys
import os
from pprint import pprint

# Add project root to path
sys.path.append(os.getcwd())

from backend.core.qdrant_store import get_qdrant_connector
from backend.core.embeddings import get_embedding_model
from backend.config import settings

def inspect_qdrant_payload():
    print("Initializing components...")
    qdrant = get_qdrant_connector()
    embedding = get_embedding_model()
    
    query = "trộm cắp tài sản"
    print(f"\nEmbedding query: {query}")
    vector = embedding.embed_query(query)
    
    print(f"\nQuerying Qdrant collection: {settings.QDRANT_LEGAL_COLLECTION}")
    response = qdrant.client.query_points(
        collection_name=settings.QDRANT_LEGAL_COLLECTION,
        query=vector.tolist(),
        using="dense",
        limit=3,
        with_payload=True  # Explicitly request payload
    )
    
    print(f"\nFound {len(response.points)} points.")
    for i, point in enumerate(response.points):
        print(f"\n--- Point {i+1} ---")
        print(f"ID: {point.id}")
        print(f"Score: {point.score}")
        print(f"Payload keys: {list(point.payload.keys()) if point.payload else 'None'}")
        if point.payload:
            pprint(point.payload)
        else:
            print("PAYLOAD IS EMPTY!")

if __name__ == "__main__":
    inspect_qdrant_payload()
