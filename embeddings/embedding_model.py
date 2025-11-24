# embeddings/embedding_model.py
import os
import json
import pickle
import yaml
import logging
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class EmbeddingModel:
    def __init__(self, model_name: str = None, config_path: str = "config/config.yaml"):
        """
        Load embedding model from HuggingFace Hub or local path.
        """
        # Load config if model_name not provided
        if model_name is None and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            model_name = cfg.get("embedding", {}).get("model_name", "BAAI/bge-large")

        if model_name is None:
            model_name = "BAAI/bge-large"

        # Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        logging.info(f"Loaded embedding model '{model_name}' on {device}")

    def encode(self, texts: list, batch_size: int = 32, convert_to_numpy=True, show_progress_bar=False):
        """
        Encode a list of texts to embeddings with batching.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", disable=not show_progress_bar):
            batch = texts[i:i+batch_size]
            batch_emb = self.model.encode(batch, convert_to_numpy=convert_to_numpy, show_progress_bar=False)
            all_embeddings.append(batch_emb)
        embeddings = np.vstack(all_embeddings)
        return embeddings

    @staticmethod
    def load_chunks(file_path: str) -> list:
        """Load chunks from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logging.info(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        logging.info(f"Saved embeddings to {file_path}")

    @staticmethod
    def save_mapping(mapping: dict, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(mapping, f)
        logging.info(f"Saved mapping to {file_path}")

    def create_embeddings(self, chunks: list, batch_size: int = 32) -> tuple[np.ndarray, dict]:
        """
        Create embeddings from a list of chunks.
        Each chunk should be a dict with at least "text" field.
        Returns embeddings array and mapping dict.
        """
        embeddings_list = []
        mapping = {}
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
            batch = chunks[i:i+batch_size]
            texts = [c["chunk"] for c in batch]
            batch_emb = self.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            embeddings_list.append(batch_emb)
            for j, c in enumerate(batch):
                mapping[i+j] = {"chunk": c["chunk"], "metadata": c.get("metadata", {})}
        embeddings = np.vstack(embeddings_list)
        logging.info(f"Created embeddings with shape {embeddings.shape}")
        return embeddings, mapping

def main():
    # --- Load config ---
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    chunks_file = cfg.get("paths", {}).get("chunks_file", "data/processed/chunks.json")
    embeddings_file = cfg.get("paths", {}).get("embeddings_file", "embeddings/bge_embeddings.npy")
    mapping_file = cfg.get("paths", {}).get("mapping_file", "faiss_index/mapping.pkl")
    batch_size = cfg.get("embedding", {}).get("batch_size", 32)
    model_name = cfg.get("embedding", {}).get("model_name", None)

    # --- Pipeline ---
    model = EmbeddingModel(model_name=model_name, config_path=config_path)
    chunks = EmbeddingModel.load_chunks(chunks_file)
    embeddings, mapping = model.create_embeddings(chunks, batch_size=batch_size)
    EmbeddingModel.save_embeddings(embeddings, embeddings_file)
    EmbeddingModel.save_mapping(mapping, mapping_file)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error occurred: {e}")
