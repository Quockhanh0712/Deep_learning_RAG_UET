# scripts/test_faiss_build.py
import os
import pickle
import numpy as np
import faiss
from embeddings.embedding_model import EmbeddingModel
from pipeline.indexing import build_faiss_index

def main():
    # Giả lập chunks nhỏ để test
    test_chunks = [
        {"chunk": "This is a test chunk 1.", "source": "test.txt", "chunk_id": 0},
        {"chunk": "Another chunk for testing.", "source": "test.txt", "chunk_id": 1},
    ]

    # Build FAISS index test
    index_path = "faiss_index/test_index.faiss"
    mapping_path = "faiss_index/test_mapping.pkl"

    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        index, mapping = build_faiss_index(test_chunks, index_path=index_path, mapping_path=mapping_path)
        print("FAISS index test created successfully!")
        print("Index:", index)
        print("Mapping:", mapping)
    except Exception as e:
        print("Error while building FAISS index:", e)

if __name__ == "__main__":
    main()
