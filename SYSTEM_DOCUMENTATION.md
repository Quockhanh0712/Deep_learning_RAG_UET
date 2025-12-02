# 📘 Tài liệu Hệ thống RAG Chatbot (BGE-M3 + ChromaDB + Gemini/Ollama)

## 1. Giới thiệu

Hệ thống **RAG Chatbot** là một giải pháp mã nguồn mở cho phép người dùng xây dựng trợ lý AI cá nhân có khả năng trả lời câu hỏi dựa trên dữ liệu riêng (PDF, TXT). Hệ thống sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)** để kết hợp sức mạnh tìm kiếm ngữ nghĩa (Semantic Search) với khả năng sinh ngữ của các mô hình ngôn ngữ lớn (LLM).

### Tính năng chính
- **Đa mô hình LLM**: Hỗ trợ linh hoạt giữa **Google Gemini** (Cloud, miễn phí/trả phí) và **Ollama** (Local, riêng tư).
- **Embedding mạnh mẽ**: Sử dụng **BAAI/bge-m3** (hoặc bge-large) cho khả năng hiểu đa ngôn ngữ và tiếng Việt vượt trội.
- **Cơ sở dữ liệu Vector**: Tích hợp **ChromaDB** để lưu trữ và truy xuất dữ liệu hiệu năng cao.
- **Giao diện đa dạng**: Cung cấp 3 tùy chọn giao diện: **Chainlit** (Chat chuyên nghiệp), **Gradio** (Web UI đơn giản), và **Streamlit** (Dashboard).
- **Tối ưu hóa GPU**: Hỗ trợ tăng tốc GPU (CUDA) và tính toán FP16 cho tốc độ xử lý nhanh.

---

## 2. Kiến trúc Hệ thống

Hệ thống hoạt động theo quy trình khép kín gồm 4 giai đoạn:

1.  **Ingestion (Nạp dữ liệu)**
    -   **Input**: File PDF hoặc TXT từ người dùng.
    -   **Processing**: Đọc nội dung -> Chia nhỏ (Chunking) thành các đoạn văn bản (mặc định 500 từ).
    -   **Embedding**: Chuyển đổi văn bản thành vector số học sử dụng mô hình BGE.
    -   **Storage**: Lưu vector và metadata vào ChromaDB.

2.  **Retrieval (Truy xuất)**
    -   **Query**: Người dùng đặt câu hỏi.
    -   **Search**: Hệ thống tìm kiếm `k` đoạn văn bản (Chunks) có nội dung tương đồng nhất với câu hỏi trong ChromaDB.

3.  **Generation (Sinh câu trả lời)**
    -   **Prompting**: Ghép câu hỏi và các đoạn văn bản tìm được vào một khuôn mẫu (Prompt).
    -   **Inference**: Gửi Prompt đến LLM (Gemini hoặc Ollama) để sinh câu trả lời.

4.  **Response (Phản hồi)**
    -   Hiển thị câu trả lời cuối cùng kèm theo nguồn tham khảo (Source citations).

---

## 3. Cài đặt và Triển khai

### Yêu cầu hệ thống
- **OS**: Windows, Linux, hoặc macOS.
- **Python**: Phiên bản 3.10 trở lên.
- **Phần cứng**:
    -   CPU: Tối thiểu 4 cores.
    -   RAM: 8GB (16GB nếu chạy Ollama local).
    -   GPU (Khuyến nghị): NVIDIA GPU với VRAM >= 4GB để tăng tốc Embedding và Ollama.

### Các bước cài đặt

1.  **Clone mã nguồn**
    ```bash
    git clone <repository_url>
    cd rag-bge-chroma-gemini
    ```

2.  **Tạo môi trường ảo (Virtual Environment)**
    ```powershell
    # Windows PowerShell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Cài đặt thư viện phụ thuộc**
    ```bash
    pip install -r requirements.txt
    ```
    *Lưu ý: Nếu dùng GPU, hãy đảm bảo đã cài đặt PyTorch phiên bản hỗ trợ CUDA.*

4.  **Cấu hình môi trường**
    Tạo file `.env` tại thư mục gốc (tham khảo mục 4).

---

## 4. Cấu hình Chi tiết (.env)

Tạo file `.env` và tùy chỉnh các tham số sau:

### Cấu hình LLM (Chọn 1 trong 2)

**Option 1: Google Gemini (Cloud)**
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIzaSy...  # API Key từ Google AI Studio
GEMINI_MODEL=gemini-1.5-flash
```

**Option 2: Ollama (Local)**
```env
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b  # Hoặc model khác đã pull về
```

### Cấu hình Embedding & Vector Store
```env
# Mô hình Embedding (Khuyên dùng BAAI/bge-m3 cho tiếng Việt)
EMBEDDING_MODEL=BAAI/bge-m3

# Cấu hình hiệu năng
EMBEDDING_BATCH_SIZE=16  # Giảm xuống 8 hoặc 4 nếu bị lỗi Out of Memory
USE_FP16=true            # True để tăng tốc trên GPU

# Đường dẫn lưu dữ liệu ChromaDB
CHROMA_PATH=./data/chroma_db
COLLECTION_NAME=documents
```

### Cấu hình RAG
```env
# Số lượng đoạn văn bản lấy làm ngữ cảnh
TOP_K=5
```

---

## 5. Hướng dẫn Sử dụng

### 5.1. Giao diện Chainlit (Khuyên dùng)
Giao diện chat hiện đại, hỗ trợ streaming và trải nghiệm người dùng tốt nhất.

- **Khởi chạy**:
  ```powershell
  chainlit run chatbot.py -w
  ```
- **Truy cập**: `http://localhost:8000`
- **Tính năng**:
  - **Upload**: Kéo thả file hoặc click icon 📎.
  - **Lệnh Chat**:
    - `/files`: Xem danh sách tài liệu.
    - `/clear`: Xóa lịch sử hội thoại.
    - `/delete <file_id>`: Xóa tài liệu cụ thể.
    - `/help`: Xem hướng dẫn.

### 5.2. Giao diện Gradio
Giao diện đơn giản, trực quan, thích hợp để demo nhanh.

- **Khởi chạy**:
  ```powershell
  python app_gradio.py
  ```
- **Truy cập**: `http://localhost:7860`
- **Tính năng**: Tab quản lý file riêng biệt, xem trước nguồn tham khảo rõ ràng.

### 5.3. Giao diện Streamlit
Giao diện dạng Dashboard, dễ dàng tùy biến layout.

- **Khởi chạy**:
  ```powershell
  streamlit run app_modern.py
  ```
- **Truy cập**: `http://localhost:8501`

---

## 6. Chi tiết Kỹ thuật (Source Code)

### `src/embeddings.py`
- Quản lý model `SentenceTransformer`.
- Tự động phát hiện GPU/CPU.
- `embed_texts`: Hàm xử lý embedding theo batch để tối ưu hiệu năng.
- `ChromaEmbeddingFunction`: Wrapper để tích hợp với ChromaDB.

### `src/file_manager.py`
- `load_file`: Đọc file PDF/TXT, chia nhỏ văn bản (chunking) và gọi hàm lưu vào DB.
- `_chunk_text`: Thuật toán chia văn bản dựa trên số lượng từ (word-based sliding window).

### `src/llm_client.py`
- Lớp trừu tượng hóa việc gọi LLM.
- Tự động chuyển đổi giữa `genai.Client` (Gemini) và `ollama.chat` dựa trên cấu hình.

### `src/vector_store.py`
- Quản lý kết nối `chromadb.PersistentClient`.
- Các hàm CRUD: `add_documents`, `query_documents`, `delete_by_file_id`, `list_files`.

---

## 7. Xử lý Sự cố (Troubleshooting)

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| **Lỗi `ModuleNotFoundError`** | Chưa kích hoạt venv hoặc thiếu thư viện | Chạy `.\.venv\Scripts\Activate.ps1` và `pip install -r requirements.txt` |
| **Lỗi `CUDA out of memory`** | GPU hết VRAM khi embedding | Giảm `EMBEDDING_BATCH_SIZE` trong `.env` xuống 8 hoặc 4. |
| **Chainlit không chạy** | Lỗi đường dẫn hoặc biến môi trường | Thử chạy `python -m chainlit run chatbot.py -w` |
| **Ollama connection refused** | Ollama chưa chạy | Mở ứng dụng Ollama hoặc chạy `ollama serve` trong terminal khác. |
| **Kết quả trả lời không liên quan** | `TOP_K` thấp hoặc dữ liệu kém | Tăng `TOP_K` lên 7-10 hoặc kiểm tra chất lượng file upload. |

---

## 8. Mở rộng & Phát triển

Để tùy chỉnh hệ thống:
1.  **Thêm định dạng file**: Sửa `src/file_manager.py` để hỗ trợ `.docx`, `.html`.
2.  **Thay đổi Prompt**: Sửa hàm `build_prompt` trong `src/rag_pipeline.py` hoặc `chatbot.py`.
3.  **Tùy chỉnh UI**:
    -   Chainlit: Sửa `.chainlit/config.toml` và `chatbot.py`.
    -   Gradio: Sửa `custom_css` trong `app_gradio.py`.

---
*Tài liệu hệ thống phiên bản 2.0 - Cập nhật ngày 02/12/2025*
