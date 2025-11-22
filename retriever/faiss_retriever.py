import faiss
import pickle
import numpy as np
from embeddings.embedding_model import EmbeddingModel

class FaissRetriever:
    def __init__(self, index_file, mapping_file, embedding_model: EmbeddingModel):
        self.index = faiss.read_index(index_file)
        with open(mapping_file, "rb") as f:
            self.mapping = pickle.load(f)
        self.model = embedding_model

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)

    def get_top_k(self, query: str, top_k: int = 5):
        q_vec = self.encode_query(query)
        distances, indices = self.index.search(q_vec, top_k)
        results = []
        for idx in indices[0]:
            if idx in self.mapping:
                results.append(self.mapping[idx])
        return results
