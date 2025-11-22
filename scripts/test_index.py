# scripts/test_index.py
import faiss
import pickle
import numpy as np
from embeddings.embedding_model import EmbeddingModel

INDEX_PATH = "faiss_index/index.faiss"
MAPPING_PATH = "faiss_index/mapping.pkl"

def main():
    # --- Load FAISS index ---
    index = faiss.read_index(INDEX_PATH)
    print(f"Loaded FAISS index from {INDEX_PATH}, total vectors = {index.ntotal}")

    # --- Load mapping ---
    with open(MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
    print(f"Loaded mapping from {MAPPING_PATH}, total items = {len(mapping)}")

    # --- Test search ---
    model = EmbeddingModel()  # load the same embedding model
    query_text = "Python basics and programming"
    query_vector = model.encode([query_text])[0].astype(np.float32)
    
    # Cosine similarity: normalize vector if index uses cosine
    faiss.normalize_L2(query_vector.reshape(1, -1))
    
    k = 5  # top-k results
    D, I = index.search(query_vector.reshape(1, -1), k)
    
    print("\nTop-k search results:")
    for rank, idx in enumerate(I[0]):
        item = mapping[idx]
        print(f"{rank+1}. chunk_id={item['chunk_id']}, source={item['source']}, text_preview={item['text'][:100]}...")

if __name__ == "__main__":
    main()
