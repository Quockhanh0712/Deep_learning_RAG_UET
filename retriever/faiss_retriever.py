import faiss
import pickle
import numpy as np
from embeddings.embedding_model import EmbeddingModel

class FaissRetriever:
    def __init__(self, index_file, mapping_file, embedding_model: EmbeddingModel):
        self.index = faiss.read_index(index_file)
        with open(mapping_file, "rb") as f:
            self.mapping = pickle.load(f)  # mapping là list
        self.model = embedding_model

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)

    def get_top_k(self, query: str, top_k: int = 5):
        # encode query
        q_vec = self.encode_query(query)
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.mapping):
                chunk = self.mapping[idx]
                # thêm distance vào chunk để debug
                chunk_info = {
                    "index": idx,
                    "distance": distances[0][i],
                    "chunk_id": chunk.get("chunk_id", idx),
                    "source": chunk.get("source", f"chunk_{idx}"),
                    "chunk": chunk.get("chunk", "")
                }
                results.append(chunk_info)
            else:
                print(f"⚠️ Warning: index {idx} out of range for mapping list")
        return results
