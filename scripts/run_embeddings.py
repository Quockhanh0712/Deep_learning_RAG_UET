# scripts/run_embeddings.py
import logging
import os
import yaml
from embeddings.embedding_model import EmbeddingModel
from pipeline.indexing import build_faiss_index  # dùng hàm trong indexing.py

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
    batch_size = cfg.get("embedding", {}).get("batch_size", 32)
    model_name = cfg.get("embedding", {}).get("model_name", None)

    # --- Load embedding model ---
    model = EmbeddingModel(model_name=model_name)

    # --- Load chunks ---
    chunks = EmbeddingModel.load_chunks(chunks_file)

    # --- Create embeddings ---
    embeddings, mapping = model.create_embeddings(chunks, batch_size=batch_size)

    # --- Save embeddings and mapping ---
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

    # --- Build FAISS index using pipeline/indexing.py ---
    build_faiss_index(chunks_for_index, index_path=index_file, mapping_path=mapping_file)

    logging.info("Embedding + FAISS index pipeline finished successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in embedding pipeline: {e}")
        logging.error(traceback.format_exc())  # <-- in full stack trace
