# pipeline/indexing.py
import os
import pickle
import yaml
import faiss
import numpy as np
from embeddings.embedding_model import EmbeddingModel
import logging

logging.basicConfig(level=logging.INFO)

# ----------------------
# Load config
# ----------------------
def load_index_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        return {
            "index_type": "Flat",
            "metric": "cosine",
            "dimension": 1024
        }
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    index_cfg = cfg.get("indexing", {})
    return {
        "index_type": index_cfg.get("index_type", "Flat"),
        "metric": index_cfg.get("metric", "cosine"),
        "dimension": index_cfg.get("dimension", 1024)
    }

INDEX_CONFIG = load_index_config()

# ----------------------
# FAISS index creation
# ----------------------
def create_faiss_index(dimension=None, metric=None):
    if dimension is None:
        dimension = INDEX_CONFIG["dimension"]
    if metric is None:
        metric = INDEX_CONFIG["metric"]

    if metric.lower() == "cosine":
        return faiss.IndexFlatIP(dimension)
    elif metric.lower() == "ip":
        return faiss.IndexFlatIP(dimension)
    else:
        return faiss.IndexFlatL2(dimension)

# ----------------------
# Build FAISS index from chunks
# ----------------------
def build_faiss_index(chunks, index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
    """
    chunks: list of dict {"chunk_id": int, "source": str, "text": str}
    """
    if not os.path.exists(os.path.dirname(index_path)):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    logging.info("Loading embedding model...")
    embedding_model = EmbeddingModel()  # mặc định config trong class
    index = create_faiss_index()
    mapping = []
    vectors = []

    logging.info("Encoding chunks...")
    for chunk in chunks:
        text = chunk["chunk"]
        vector = embedding_model.encode([text])[0].astype("float32")
        # normalize nếu cosine
        if INDEX_CONFIG["metric"].lower() == "cosine":
            faiss.normalize_L2(vector.reshape(1, -1))
        vectors.append(vector)
        mapping.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk.get("source", "unknown"),
            "chunk": text,
            "embedding": vector  # lưu embedding để đồng bộ
        })

    vectors = np.vstack(vectors)
    index.add(vectors)

    # Save index and mapping
    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(mapping, f)

    logging.info(f"FAISS index saved to {index_path}, mapping saved to {mapping_path}")
    return index, mapping

# ----------------------
# Load existing FAISS index
# ----------------------
def load_faiss_index(index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError("FAISS index or mapping file not found")

    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    return index, mapping

# ----------------------
# Update index with new chunks
# ----------------------
def update_index(new_chunks, index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
    index, mapping = load_faiss_index(index_path, mapping_path)
    embedding_model = EmbeddingModel()
    vectors = []

    logging.info("Encoding new chunks...")
    for chunk in new_chunks:
        text = chunk["chunk"]
        vector = embedding_model.encode([text])[0].astype("float32")
        if INDEX_CONFIG["metric"].lower() == "cosine":
            faiss.normalize_L2(vector.reshape(1, -1))
        vectors.append(vector)
        mapping.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk.get("source", "unknown"),
            "chunk": text,
            "embedding": vector
        })

    vectors = np.vstack(vectors)
    index.add(vectors)

    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(mapping, f)

    logging.info(f"FAISS index updated with {len(new_chunks)} new chunks.")
    return index, mapping


##########################################################################################333\

# # pipeline/indexing.py
# import os
# import pickle
# import yaml
# import faiss
# import numpy as np
# from embeddings.embedding_model import EmbeddingModel
# import logging

# logging.basicConfig(level=logging.INFO)

# def load_index_config(config_path="config/config.yaml"):
#     """
#     Load indexing parameters from config.yaml
#     """
#     if not os.path.exists(config_path):
#         return {
#             "index_type": "Flat",
#             "metric": "cosine"
#         }
#     with open(config_path, "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)
#     index_cfg = cfg.get("indexing", {})
#     return {
#         "index_type": index_cfg.get("index_type", "Flat"),
#         "metric": index_cfg.get("metric", "cosine")
#     }

# INDEX_CONFIG = load_index_config()

# def create_faiss_index(dim, metric="cosine"):
#     """
#     Create FAISS index with given dimension and metric
#     """
#     if metric.lower() == "cosine" or metric.lower() == "ip":
#         return faiss.IndexFlatIP(dim)
#     else:
#         return faiss.IndexFlatL2(dim)

# def build_faiss_index(chunks, index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
#     """
#     Build FAISS index from chunks list. Each chunk is a dict:
#     {"chunk": str, "source": str, "chunk_id": int}
#     """
#     if not os.path.exists(os.path.dirname(index_path)):
#         os.makedirs(os.path.dirname(index_path), exist_ok=True)

#     # Load model and get dimension
#     embedding_model = EmbeddingModel()
#     dim = embedding_model.dim
#     index = create_faiss_index(dim, INDEX_CONFIG["metric"])

#     vectors = []
#     mapping = []

#     texts = [chunk["chunk"] for chunk in chunks]
#     embeddings = embedding_model.encode(texts, batch_size=32)
#     embeddings = np.array(embeddings, dtype=np.float32)

#     # Normalize if cosine
#     if INDEX_CONFIG["metric"].lower() == "cosine":
#         faiss.normalize_L2(embeddings)

#     vectors = embeddings
#     index.add(vectors)

#     # Build mapping
#     for i, chunk in enumerate(chunks):
#         mapping.append({
#             "chunk_id": chunk["chunk_id"],
#             "source": chunk.get("source", "unknown"),
#             "text": chunk["chunk"]
#         })

#     # Save index and mapping
#     faiss.write_index(index, index_path)
#     with open(mapping_path, "wb") as f:
#         pickle.dump(mapping, f)

#     logging.info(f"FAISS index saved to {index_path}, mapping saved to {mapping_path}")
#     return index, mapping

# def load_faiss_index(index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
#     """
#     Load FAISS index and mapping
#     """
#     if not os.path.exists(index_path) or not os.path.exists(mapping_path):
#         raise FileNotFoundError("Index or mapping file not found")

#     index = faiss.read_index(index_path)
#     with open(mapping_path, "rb") as f:
#         mapping = pickle.load(f)
#     return index, mapping

# def update_faiss_index(new_chunks, index_path="faiss_index/index.faiss", mapping_path="faiss_index/mapping.pkl"):
#     """
#     Add new chunks to existing FAISS index, avoid duplicate chunk_id
#     """
#     index, mapping = load_faiss_index(index_path, mapping_path)
#     embedding_model = EmbeddingModel()
#     dim = embedding_model.dim

#     # Filter out duplicates
#     existing_ids = set(m["chunk_id"] for m in mapping)
#     new_chunks = [c for c in new_chunks if c["chunk_id"] not in existing_ids]

#     if not new_chunks:
#         logging.info("No new chunks to add.")
#         return index, mapping

#     texts = [c["chunk"] for c in new_chunks]
#     embeddings = embedding_model.encode(texts, batch_size=32)
#     embeddings = np.array(embeddings, dtype=np.float32)
#     if INDEX_CONFIG["metric"].lower() == "cosine":
#         faiss.normalize_L2(embeddings)

#     index.add(embeddings)

#     for c in new_chunks:
#         mapping.append({
#             "chunk_id": c["chunk_id"],
#             "source": c.get("source", "unknown"),
#             "text": c["chunk"]
#         })

#     faiss.write_index(index, index_path)
#     with open(mapping_path, "wb") as f:
#         pickle.dump(mapping, f)

#     logging.info(f"FAISS index updated with {len(new_chunks)} new chunks.")
#     return index, mapping
