# RAG QA System - Full Run Guide

Hướng dẫn chạy **pipeline** từ đầu với các file thực tế của bạn.

---

## 1. Tạo môi trường ảo và cài dependencies

```powershell
# Tạo virtualenv
python -m venv venv

# Activate
.\venv\Scripts\activate

# Update pip
pip install --upgrade pip

# Cài requirements
pip install -r requirements.txt
2. Chuẩn bị dữ liệu mới
Tạo thư mục documents:

powershell
Copy code
mkdir data\documents
Thêm file txt của dataset mới vào data\documents\
Ví dụ: doc1.txt, doc2.txt, ...

3. Chạy pipeline tạo embeddings
powershell
Copy code
python pipeline/create_embeddings.py --input_dir data/documents --output_file data/documents.pkl
create_embeddings.py đọc tất cả text trong data/documents/ và tính embedding bằng EmbeddingModel.

Lưu ra documents.pkl.

4. (Tuỳ chọn) Build FAISS index
powershell
Copy code
python pipeline/build_faiss_index.py --embeddings_file data/documents.pkl --output_index data/faiss_index.idx
build_faiss_index.py đọc embeddings từ .pkl và tạo FAISS index để tìm kiếm nhanh.

5. Chạy RAG pipeline cho câu hỏi đơn lẻ
powershell
Copy code
python pipeline/run_rag_pipeline.py --query "Python là gì?" --top_k 5
run_rag_pipeline.py load embeddings, FAISS (nếu có), LLMWrapper và trả về answer + top chunks.

--top_k xác định số chunk hiển thị.

Ví dụ output:

bash
Copy code
Answer: Python là ngôn ngữ lập trình cao cấp...
Top chunks: ["Chunk 1 text...", "Chunk 2 text..."]
6. Chạy multi-question evaluation
Tạo evaluation/test_questions.json:

json
Copy code
[
    "Python là gì?",
    "Làm thế nào để khai báo biến trong Python?",
    "Các kiểu dữ liệu cơ bản trong Python là gì?"
]
(Tuỳ chọn) Tạo evaluation/ground_truth.json:

json
Copy code
{
    "Q1": "Python is a high-level programming language...",
    "Q2": "Variables can be declared using = operator...",
    "Q3": "Basic types: int, float, str, bool..."
}
Chạy evaluation:

powershell
Copy code
python -m evaluation.evaluate
Tính EM, F1, cosine similarity.

In ra kết quả từng câu hỏi + điểm trung bình.

7. Chạy Streamlit UI
powershell
Copy code
cd app
streamlit run streamlit_app.py
Giao diện web để hỏi RAG trực tiếp.

Có thể upload thêm documents mới.

Logs lưu ở logs/query_logs.json.

Notes
Embedding model: sử dụng SentenceTransformer (BAAI/bge-large-en-v1.5).

LLM model: local model hoặc Hugging Face, phải hỗ trợ .generate(prompt).

Pipeline files:

pipeline/create_embeddings.py

pipeline/build_faiss_index.py

pipeline/run_rag_pipeline.py

FAISS chỉ cần khi dataset lớn, giúp search nhanh.

LLMWrapper hỗ trợ multi-turn chat qua .chat(history).

css
Copy code

---

Nếu bạn muốn, tôi có thể viết **phiên bản README cực “ready-to-run”**, tất cả **file pipeline `.py` cần tạo**, với **mã lệnh sẵn sàng copy** để người dùng chỉ cần `git clone` + `venv` + `pip install` + chạy từng file theo thứ tự, không phải sửa gì cả.  








