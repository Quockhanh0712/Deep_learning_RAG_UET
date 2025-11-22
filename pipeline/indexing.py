# pipeline/indexing.py
import os
import pickle
import yaml
import faiss
import numpy as np
from embeddings.embedding_model import EmbeddingModel  # giả sử bạn có class này

def load_index_config(config_path="config/config.yaml"):
    """
    Load indexing parameters from config.yaml
    """
    if not os.path.exists(config_path):
        return {
            "index_type": "Flat",
            "metric": "cosine",
            "dimension": 6144
        }  # default
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    index_cfg = cfg.get("indexing", {})
    return {
        "index_type": index_cfg.get("index_type", "Flat"),
        "metric": index_cfg.get("metric", "cosine"),
        "dimension": index_cfg.get("dimension", 6144)
    }

INDEX_CONFIG = load_index_config()

def create_faiss_index(dimension=None, metric=None):
    """
    Create a FAISS index based on type and metric
    """
    if dimension is None:
        dimension = INDEX_CONFIG["dimension"]
    if metric is None:
        metric = INDEX_CONFIG["metric"]

    if metric.lower() == "cosine":
        # FAISS cosine = normalized vectors + inner product
        index = faiss.IndexFlatIP(dimension)
    elif metric.lower() == "ip":
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)  # default L2
    return index

def build_faiss_index(chunks: list, index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
    """
    Build FAISS index from list of chunks.
    Each chunk is a dict: {"chunk": str, "source": str, "chunk_id": int}
    """
    if not os.path.exists(os.path.dirname(index_path)):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    embedding_model = EmbeddingModel()  # load model
    index = create_faiss_index()
    mapping = []

    vectors = []
    for chunk in chunks:
        text = chunk["chunk"]
        vector = embedding_model.encode([text])[0]  # giả sử trả về np.array
        vector = np.array(vector, dtype=np.float32)
        # nếu dùng cosine, normalize
        if INDEX_CONFIG["metric"].lower() == "cosine":
            faiss.normalize_L2(vector.reshape(1, -1))
        vectors.append(vector)
        mapping.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk.get("source", "unknown"),
            "text": text
        })

    vectors = np.vstack(vectors)
    index.add(vectors)

    # Lưu index
    faiss.write_index(index, index_path)
    # Lưu mapping
    with open(mapping_path, "wb") as f:
        pickle.dump(mapping, f)

    print(f"FAISS index saved to {index_path}, mapping saved to {mapping_path}")
    return index, mapping

def load_faiss_index(index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
    """
    Load FAISS index and mapping
    """
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError("Index or mapping file not found")

    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    return index, mapping

def update_index(new_chunks: list, index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
    """
    Add new chunks to existing FAISS index
    """
    index, mapping = load_faiss_index(index_path, mapping_path)
    embedding_model = EmbeddingModel()

    vectors = []
    for chunk in new_chunks:
        text = chunk["chunk"]
        vector = embedding_model.encode([text])[0]
        vector = np.array(vector, dtype=np.float32)
        if INDEX_CONFIG["metric"].lower() == "cosine":
            faiss.normalize_L2(vector.reshape(1, -1))
        vectors.append(vector)
        mapping.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk.get("source", "unknown"),
            "text": text
        })

    vectors = np.vstack(vectors)
    index.add(vectors)

    # Save updated index and mapping
    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(mapping, f)

    print(f"FAISS index updated with {len(new_chunks)} new chunks.")
    return index, mapping
