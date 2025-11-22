import json
import yaml
import numpy as np
from pipeline.rag_pipeline import RAGPipeline
from sklearn.metrics import f1_score
from embeddings.embedding_model import EmbeddingModel

# --- Load JSON ---
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Metrics ---
def exact_match(pred, truth):
    return float(pred.strip() == truth.strip())

def f1_token(pred, truth):
    pred_tokens = pred.strip().split()
    truth_tokens = truth.strip().split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def cosine_similarity(pred, truth, embedding_model: EmbeddingModel):
    vec_pred = embedding_model.encode([pred], convert_to_numpy=True)[0]
    vec_truth = embedding_model.encode([truth], convert_to_numpy=True)[0]
    vec_pred /= np.linalg.norm(vec_pred)
    vec_truth /= np.linalg.norm(vec_truth)
    return float(np.dot(vec_pred, vec_truth))

# --- Run evaluation ---
def run_evaluation(rag_pipeline, test_questions, ground_truth=None, embedding_model=None, weights=None):
    """
    Chạy evaluation trên RAG pipeline.

    test_questions: list các câu hỏi (hoặc dict {id: question})
    ground_truth: dict {id: answer} (có thể None nếu không có)
    weights: dict, ví dụ {"em":0.4, "f1":0.3, "cosine":0.3}
    """
    if weights is None:
        weights = {"em": 0.4, "f1": 0.3, "cosine": 0.3}
        
    # Nếu test_questions là list, chuyển thành dict với key là string "1", "2", ...
    if isinstance(test_questions, list):
        test_questions = {str(i+1): q for i, q in enumerate(test_questions)}
    
    if ground_truth is None:
        ground_truth = {}

    results = []
    for qid, question in test_questions.items():
        pred = rag_pipeline.run(question)
        truth = ground_truth.get(qid, "")
        
        em_score = exact_match(pred, truth)
        f1_score_val = f1_token(pred, truth)
        cosine_score_val = cosine_similarity(pred, truth, embedding_model) if embedding_model else 0.0
        
        total_score = (
            weights.get("em",0)*em_score +
            weights.get("f1",0)*f1_score_val +
            weights.get("cosine",0)*cosine_score_val
        )
        
        results.append({
            "qid": qid,
            "question": question,
            "prediction": pred,
            "ground_truth": truth,
            "em": em_score,
            "f1": f1_score_val,
            "cosine": cosine_score_val,
            "score": total_score
        })
        
        print(f"Q{qid}: {question}")
        print(f"A{qid}: {pred}\n")
        
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"Average combined score: {avg_score:.4f}")
    return results


# --- Main ---
if __name__ == "__main__":

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_questions = load_json("evaluation/test_questions.json")
    ground_truth = load_json("evaluation/ground_truth.json")
    
    # Initialize embedding model for cosine similarity
    embedding_model = EmbeddingModel(cfg["embedding"]["model_name"])
    
    rag_pipeline = RAGPipeline(cfg)
    results = run_evaluation(rag_pipeline, test_questions, ground_truth, embedding_model=embedding_model)
    
    # Save results
    with open("evaluation/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
