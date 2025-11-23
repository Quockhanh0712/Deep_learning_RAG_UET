import numpy as np
from embeddings.embedding_model import EmbeddingModel
import logging

logging.basicConfig(level=logging.INFO)

def cosine_similarity(vec1, vec2):
    """
    vec1: (dim,) hoặc (num_queries, dim)
    vec2: (num_docs, dim)
    Trả về mảng (num_queries, num_docs) nếu vec1 là 2D, hoặc (num_docs,) nếu vec1 là 1D
    """
    if vec1.ndim == 1:
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
        return np.dot(vec2_norm, vec1)  # (num_docs,)
    else:
        vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
        return vec1_norm @ vec2_norm.T   # (num_queries, num_docs)

def inner_product(vec1, vec2):
    if vec1.ndim == 1:
        return vec2 @ vec1
    else:
        return vec1 @ vec2.T

def l2_distance(vec1, vec2):
    if vec1.ndim == 1:
        return np.linalg.norm(vec2 - vec1, axis=1)
    else:
        # vec1: (num_queries, dim), vec2: (num_docs, dim)
        return np.sqrt(((vec1[:, None, :] - vec2[None, :, :]) ** 2).sum(axis=2))

def main():
    # --- Load embeddings ---
    text_emb_file = "embeddings/bge_embeddings.npy"
    question_emb_file = "embeddings/bge_question_embeddings.npy"

    text_emb = np.load(text_emb_file)       # (num_docs, dim)
    question_emb = np.load(question_emb_file)  # (num_queries, dim) hoặc (dim,)

    # --- Fix shape nếu nhầm ---
    if question_emb.ndim == 1:
        question_emb = question_emb[None, :]  # (1, dim)
    elif question_emb.shape[1] != text_emb.shape[1]:
        logging.warning(f"Transposing question_emb: {question_emb.shape} -> ", end="")
        question_emb = question_emb.T
        logging.warning(f"{question_emb.shape}")

    logging.info(f"text_emb shape: {text_emb.shape}")
    logging.info(f"question_emb shape: {question_emb.shape}")

    # --- Cosine similarity ---
    cos_sim = cosine_similarity(question_emb, text_emb)
    logging.info(f"Cosine similarity shape: {cos_sim.shape}")

    # --- Inner product ---
    ip_sim = inner_product(question_emb, text_emb)
    logging.info(f"Inner product shape: {ip_sim.shape}")

    # --- L2 distance ---
    l2_sim = l2_distance(question_emb, text_emb)
    logging.info(f"L2 distance shape: {l2_sim.shape}")

    # --- Example: top-5 docs for first query ---
    first_query_sim = cos_sim[0]  # luôn lấy query đầu tiên
    top5_idx = np.argsort(-first_query_sim)[:5]
    print("Top 5 docs for first query (by cosine similarity):", top5_idx)
    print("Scores:", first_query_sim[top5_idx])

if __name__ == "__main__":
    main()
