# scripts/debug_faiss_mapping.py
import faiss
import pickle
import logging
from embeddings.embedding_model import EmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Config paths ---
INDEX_FILE = "faiss_index/index.faiss"
MAPPING_FILE = "faiss_index/mapping.pkl"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

def main():
    # Load FAISS index
    logging.info(f"Loading FAISS index from {INDEX_FILE}")
    index = faiss.read_index(INDEX_FILE)
    logging.info(f"FAISS index dimension: {index.d}, nb vectors: {index.ntotal}")

    # Load mapping
    logging.info(f"Loading mapping from {MAPPING_FILE}")
    with open(MAPPING_FILE, "rb") as f:
        mapping = pickle.load(f)

    # Xác định type mapping
    if isinstance(mapping, dict):
        logging.info(f"Mapping type: dict, size: {len(mapping)}")
        sample_keys = list(mapping.keys())[:10]
    elif isinstance(mapping, list):
        logging.info(f"Mapping type: list, size: {len(mapping)}")
        sample_keys = list(range(min(10, len(mapping))))
    else:
        raise TypeError(f"Unexpected mapping type: {type(mapping)}")

    logging.info(f"Mapping sample keys/indices: {sample_keys}")

    # Load embedding model
    logging.info(f"Loading embedding model {EMBEDDING_MODEL_NAME}")
    embed_model = EmbeddingModel(EMBEDDING_MODEL_NAME)

    # Test query
    test_query = "How should crimes committed by a person be handled according to the law?"
    logging.info(f"Encoding test query: {test_query}")
    q_vec = embed_model.encode([test_query], convert_to_numpy=True)
    logging.info(f"Query vector shape: {q_vec.shape}")

    # Search FAISS
    top_k = 5
    distances, indices = index.search(q_vec, top_k)
    logging.info(f"FAISS distances: {distances}")
    logging.info(f"FAISS indices: {indices}")

    # Check mapping for retrieved indices
    retrieved_chunks = []
    for idx in indices[0]:
        if isinstance(mapping, dict):
            if idx in mapping:
                retrieved_chunks.append(mapping[idx])
            else:
                logging.warning(f"Index {idx} not found in mapping!")
        elif isinstance(mapping, list):
            if 0 <= idx < len(mapping):
                retrieved_chunks.append(mapping[idx])
            else:
                logging.warning(f"Index {idx} out of range for list mapping!")

    logging.info(f"Number of retrieved chunks: {len(retrieved_chunks)}")
    for i, chunk in enumerate(retrieved_chunks, 1):
        text_preview = getattr(chunk, "text", str(chunk))[:200]
        logging.info(f"[Chunk {i}] {text_preview}...")

if __name__ == "__main__":
    main()
