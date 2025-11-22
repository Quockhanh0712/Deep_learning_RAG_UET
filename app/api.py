from fastapi import FastAPI
from pydantic import BaseModel
import yaml
from pipeline.rag_pipeline import RAGPipeline

# --- Load config ---
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# --- Initialize pipeline ---
rag_pipeline = RAGPipeline(cfg)

# --- FastAPI app ---
app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.query
    answer = rag_pipeline.run(query)
    # Nếu muốn có top-k chunks:
    # chunks = rag_pipeline.get_top_k_chunks(query)
    return {
        "query": query,
        "answer": answer,
        # "chunks": chunks
    }
