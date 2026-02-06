from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
result = client.scroll(
    collection_name="legal_rag_hybrid",
    limit=1,
    with_payload=True,
    with_vectors=False
)

point = result[0][0]
print("PAYLOAD KEYS:", list(point.payload.keys()))
print()
for k, v in point.payload.items():
    print(f"[{k}]: {str(v)[:200]}")
