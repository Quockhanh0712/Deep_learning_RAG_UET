# scripts/run_embeddings.py
import logging
import os
import yaml
import json
import traceback
from embeddings.embedding_model import EmbeddingModel
from pipeline.indexing import build_faiss_index
from pipeline.chunking import process_file  # dùng hàm từ chunking.py

def load_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    logging.basicConfig(level=logging.INFO)
    
    cfg = load_config()
    
    # --- Paths ---
    chunks_file = cfg.get("paths", {}).get("chunks_file", "data/processed/chunks.json")
    embeddings_file = cfg.get("paths", {}).get("embeddings_file", "embeddings/bge_embeddings.npy")
    mapping_file = cfg.get("paths", {}).get("mapping_file", "faiss_index/mapping.pkl")
    index_file = cfg.get("paths", {}).get("index_file", "faiss_index/index.faiss")
    raw_file = cfg.get("paths", {}).get("raw_file", "data/raw/book.txt")
    batch_size = cfg.get("embedding", {}).get("batch_size", 32)
    model_name = cfg.get("embedding", {}).get("model_name", None)

    # --- Tạo chunks nếu chưa có ---
    if not os.path.exists(chunks_file):
        logging.info(f"{chunks_file} not found, creating chunks from {raw_file} ...")
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"{raw_file} not found, cannot create chunks")
        chunks = process_file(raw_file)
        os.makedirs(os.path.dirname(chunks_file), exist_ok=True)
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(chunks)} chunks to {chunks_file}")
    else:
        chunks = EmbeddingModel.load_chunks(chunks_file)

    # --- Load embedding model ---
    model = EmbeddingModel(model_name=model_name)

    # --- Create embeddings ---
    embeddings, mapping = model.create_embeddings(chunks, batch_size=batch_size)

    # --- Save embeddings and mapping ---
    os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    EmbeddingModel.save_embeddings(embeddings, embeddings_file)
    EmbeddingModel.save_mapping(mapping, mapping_file)

    # --- Convert mapping dict to list of chunks for FAISS ---
    chunks_for_index = []
    for i, m in mapping.items():
        chunks_for_index.append({
            "chunk": m["text"],
            "source": m.get("metadata", {}).get("source", f"chunk_{i}"),
            "chunk_id": i
        })

    # --- Build FAISS index ---
    build_faiss_index(chunks_for_index, index_path=index_file, mapping_path=mapping_file)

    logging.info("Embedding + FAISS index pipeline finished successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in embedding pipeline: {e}")
        logging.error(traceback.format_exc())
