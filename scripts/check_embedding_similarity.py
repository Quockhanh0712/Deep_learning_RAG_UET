import numpy as np
from embeddings.embedding_model import EmbeddingModel
import logging

logging.basicConfig(level=logging.INFO)

def compute_cosine_similarity(text, question, model_name="BAAI/bge-large-en-v1.5"):
    """
    text: str hoặc list[str]
    question: str hoặc list[str]
    model_name: tên model embedding
    Trả về: cosine similarity (scalar nếu 1 text, 1 question; mảng nếu nhiều text hoặc nhiều question)
    """
    # --- Load model ---
    model = EmbeddingModel(model_name=model_name)
    
    # --- Encode ---
    text_emb = model.encode(text)
    question_emb = model.encode(question)

    # --- Chuyển sang 2D nếu 1D ---
    if text_emb.ndim == 1:
        text_emb = text_emb.reshape(1, -1)
    if question_emb.ndim == 1:
        question_emb = question_emb.reshape(1, -1)

    # --- Chuẩn hóa ---
    text_norm = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    question_norm = question_emb / np.linalg.norm(question_emb, axis=1, keepdims=True)

    # --- Cosine similarity ---
    cos_sim = question_norm @ text_norm.T  # (num_questions, num_texts)
    
    # Nếu chỉ 1 text và 1 question, trả về scalar
    if cos_sim.shape == (1,1):
        return float(cos_sim[0,0])
    return cos_sim

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    text = "Hawkeye is the main protagonist of The Last of the Mohicans."
    question = "Who is the main protagonist of the story?"
    sim = compute_cosine_similarity(text, question)
    print("Cosine similarity:", sim)
