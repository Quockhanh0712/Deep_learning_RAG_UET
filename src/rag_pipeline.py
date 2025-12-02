import os
from typing import List, Dict

from src.vector_store import query_documents
from src.llm_client import generate_answer

TOP_K = int(os.getenv("TOP_K", "5"))

def build_prompt(question: str, docs: List[str]) -> str:
    context = "\n\n".join(docs)
    prompt = (
        "Bạn là trợ lý AI. Trả lời ngắn gọn, chính xác dựa trên ngữ cảnh sau.\n"
        "Nếu không tìm thấy thông tin phù hợp, hãy nói là không biết.\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời:"
    )
    return prompt

def rag_answer(question: str, k: int | None = None) -> Dict:
    top_k = k or TOP_K

    print(f"[RAG] Question: {question!r}")
    res = query_documents(question, k=top_k)
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    print(f"[RAG] Retrieved docs: {len(docs)}")
    prompt = build_prompt(question, docs)
    answer = generate_answer(prompt)
    print(f"[RAG] Answer length: {len(answer or '')}")
    return {
        "answer": answer,
        "context": docs,
        "metadatas": metadatas,
    }
