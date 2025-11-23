import yaml
import logging
import pickle
import faiss
import numpy as np
from embeddings.embedding_model import EmbeddingModel
from retriever.faiss_retriever import FaissRetriever

logging.basicConfig(level=logging.INFO)

# ----------------------
# Load config
# ----------------------
cfg_path = "config/config.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

paths = cfg["paths"]
embedding_model_name = cfg["embedding"]["model_name"]
top_k = cfg.get("indexing", {}).get("top_k", 5)
dim = cfg.get("indexing", {}).get("dimension", 1024)

# ----------------------
# Load embedding model
# ----------------------
embedding_model = EmbeddingModel(model_name=embedding_model_name)
logging.info(f"Loaded embedding model: {embedding_model_name}")

# ----------------------
# Load mapping
# ----------------------
mapping_file = paths["mapping_file"]
with open(mapping_file, "rb") as f:
    mapping = pickle.load(f)

logging.info(f"Số lượng chunks trong mapping: {len(mapping)}")
logging.info(f"Chunk đầu tiên keys: {mapping[0].keys()}")

# ----------------------
# Rebuild FAISS index nếu mapping chưa có embedding
# ----------------------

import pickle
import numpy as np

with open("faiss_index/mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

count_valid = 0
for c in mapping:
    emb = c.get("embedding")
    if emb is not None and isinstance(emb, (np.ndarray, list)) and len(emb) > 0:
        count_valid += 1

print(f"Số chunk có embedding hợp lệ: {count_valid} / {len(mapping)}")



needs_rebuild = any("embedding" not in chunk or chunk["embedding"] is None for chunk in mapping)

if needs_rebuild:
    logging.info("Encoding chunks embeddings and rebuilding FAISS index...")
    chunks_texts = [chunk["text"] for chunk in mapping]
    chunks_emb = embedding_model.encode(chunks_texts)
    chunks_emb = np.array(chunks_emb, dtype="float32")
    # Lưu embedding vào mapping để lần sau dùng luôn
    for i, chunk in enumerate(mapping):
        chunk["embedding"] = chunks_emb[i]
    # Lưu lại mapping.pkl
    with open(mapping_file, "wb") as f:
        pickle.dump(mapping, f)
    # Rebuild FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(chunks_emb)
    faiss.write_index(index, paths["faiss_index"])
    logging.info(f"Rebuilt FAISS index with {index.ntotal} vectors.")
else:
    # Nếu mapping đã có embedding, chỉ load index
    index = faiss.read_index(paths["faiss_index"])
    logging.info(f"Loaded FAISS index with {index.ntotal} vectors.")

# ----------------------
# Tạo retriever
# ----------------------
retriever = FaissRetriever(
    index_file=paths["faiss_index"],
    mapping_file=paths["mapping_file"],
    embedding_model=embedding_model
)

logging.info(f"FAISS index dimension: {retriever.index.d}")
logging.info(f"Number of vectors in index: {retriever.index.ntotal}")

# ----------------------
# Queries
# ----------------------
query_list = [
    "Who is the main protagonist of The Last of the Mohicans?",
    "Who is the main antagonist?"
]

for i, query in enumerate(query_list, 1):
    print(f"\nQuery {i}: {query}")
    top_chunks = retriever.get_top_k(query, top_k=top_k)

    if not top_chunks:
        print("No chunks retrieved! Check mapping.pkl and FAISS index consistency.")
        continue

    for rank, chunk in enumerate(top_chunks, 1):
        print(f"{rank}. Source: {chunk.get('source', 'N/A')}")
        print(f"   Snippet: {chunk.get('text', 'N/A')[:200]}")
        print(f"   Score: {chunk.get('score', 'N/A')}")
