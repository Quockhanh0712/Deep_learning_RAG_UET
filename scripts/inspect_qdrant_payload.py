"""
Script to inspect actual Qdrant payload structure.
This will help identify the correct key names for text content.
"""
import sys
import os
sys.path.append(os.getcwd())

from qdrant_client import QdrantClient

def inspect_qdrant_data():
    print("=" * 60)
    print("QDRANT PAYLOAD INSPECTOR")
    print("=" * 60)
    
    # Direct connection
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "legal_rag_hybrid"
    
    print(f"\nCollection: {collection_name}")
    
    # Get collection info
    try:
        info = client.get_collection(collection_name)
        print(f"Total points: {info.points_count}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Scroll to get sample points
    print("\n" + "-" * 60)
    print("SAMPLE POINTS:")
    print("-" * 60)
    
    results = client.scroll(
        collection_name=collection_name,
        limit=2,
        with_payload=True,
        with_vectors=False
    )
    
    points = results[0]
    
    for i, point in enumerate(points):
        print(f"\n=== Point {i+1} (ID: {point.id}) ===")
        if point.payload:
            print(f"PAYLOAD KEYS: {list(point.payload.keys())}")
            print()
            for key, value in point.payload.items():
                str_value = str(value)
                if len(str_value) > 200:
                    str_value = str_value[:200] + "..."
                print(f"[{key}]: {str_value}")
        else:
            print("NO PAYLOAD!")

if __name__ == "__main__":
    inspect_qdrant_data()
