import json
from pipeline.logger import log_query
from pipeline.rag_pipeline import RAGPipeline  # assuming bạn có class này
from embeddings.embedding_model import EmbeddingModel

# --- Load config và init pipeline ---
import yaml, os
def load_config(path="config/config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
rag = RAGPipeline(cfg)

# --- Load test questions và ground truth ---
questions_file = "evaluation/test_questions.json"
ground_truth_file = "evaluation/ground_truth.json"

with open(questions_file, "r", encoding="utf-8") as f:
    questions = json.load(f)

with open(ground_truth_file, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

results = []

for i, q in enumerate(questions, 1):
    answer = rag.run(q)
    gt = ground_truth.get(q, "")
    
    em = 1.0 if answer.strip().lower() == gt.strip().lower() else 0.0
    
    # Tính F1 đơn giản
    answer_tokens = set(answer.lower().split())
    gt_tokens = set(gt.lower().split())
    if not answer_tokens and not gt_tokens:
        f1 = 1.0
    elif not answer_tokens or not gt_tokens:
        f1 = 0.0
    else:
        common = answer_tokens & gt_tokens
        precision = len(common) / len(answer_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results.append({
        "qid": str(i),
        "question": q,
        "prediction": answer,
        "ground_truth": gt,
        "em": em,
        "f1": f1
    })

# --- Lưu kết quả ---
output_file = "data/test_rag_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved test results to {output_file}")
