# 🤖 RAG Chatbot - Hướng dẫn giao diện

Dự án cung cấp **3 giao diện chatbot** chuyên nghiệp để bạn lựa chọn:

## 📊 So sánh các giao diện

| Tính năng | Streamlit Modern | Gradio | Chainlit |
|-----------|------------------|--------|----------|
| Tốc độ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Giao diện | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Dễ customize | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Chat experience | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| File upload | ✅ | ✅ | ✅ |
| Streaming | ❌ | ✅ | ✅ |

---

## 1️⃣ Streamlit Modern (`app_modern.py`)

Giao diện Streamlit được nâng cấp với CSS hiện đại.

### Chạy:
```bash
streamlit run app_modern.py
```

### Tính năng:
- 🎨 Giao diện gradient đẹp mắt
- 💬 Chat interface với message bubbles
- 📚 Sidebar quản lý tài liệu
- 🔍 Hiển thị nguồn tham khảo

---

## 2️⃣ Gradio (`app_gradio.py`)

Giao diện sử dụng Gradio - thư viện phổ biến cho AI/ML demos.

### Chạy:
```bash
python app_gradio.py
```

Hoặc:
```bash
gradio app_gradio.py
```

### Tính năng:
- ⚡ Tốc độ cao, responsive
- 🎨 Theme tùy chỉnh đẹp
- 📤 Drag & drop file upload
- 📖 Tab hiển thị nguồn tham khảo
- 🔄 Auto-refresh danh sách file

### Truy cập:
- Local: http://localhost:7860
- Share: Có thể bật share mode

---

## 3️⃣ Chainlit (`chatbot.py`) ⭐ Khuyến nghị

Giao diện chuyên nghiệp nhất, tối ưu cho chatbot AI.

### Chạy:
```bash
chainlit run chatbot.py -w
```

Flag `-w` để hot reload khi thay đổi code.

### Tính năng:
- 🚀 Tốc độ cực nhanh
- 💬 Streaming responses
- 📎 Drag & drop file upload
- 📖 Context panel bên cạnh
- 🎨 Giao diện pro như ChatGPT
- 🔧 Dễ dàng config qua `.chainlit/config.toml`

### Commands đặc biệt:
- `/files` - Xem danh sách tài liệu
- `/clear` - Xóa lịch sử chat
- `/delete <file_id>` - Xóa tài liệu

### Truy cập:
- Local: http://localhost:8000

---

## 🔧 Cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc cài riêng từng thư viện
pip install chainlit gradio
```

---

## 🎨 Tùy chỉnh giao diện

### Chainlit
Chỉnh sửa file `.chainlit/config.toml`:
```toml
[UI]
name = "My Chatbot"
description = "Mô tả của bạn"
custom_css = "..."
```

### Gradio
Thay đổi theme trong `app_gradio.py`:
```python
theme=gr.themes.Soft(
    primary_hue="blue",  # Màu chính
    secondary_hue="cyan",  # Màu phụ
)
```

### Streamlit
Chỉnh CSS trong `app_modern.py`:
```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(...);
    }
</style>
""", unsafe_allow_html=True)
```

---

## 📁 Cấu trúc file

```
├── app.py              # Giao diện Streamlit cơ bản (gốc)
├── app_modern.py       # Giao diện Streamlit nâng cấp
├── app_gradio.py       # Giao diện Gradio
├── chatbot.py          # Giao diện Chainlit
├── .chainlit/
│   └── config.toml     # Config cho Chainlit
└── CHATBOT_UI_GUIDE.md # File hướng dẫn này
```

---

## 💡 Khuyến nghị sử dụng

| Use case | Giao diện khuyến nghị |
|----------|----------------------|
| Production, demo khách hàng | **Chainlit** |
| Prototype nhanh | Gradio |
| Tùy chỉnh sâu UI | Streamlit Modern |
| Chia sẻ public | Gradio (share mode) |

---

## 🚀 Quick Start

```bash
# Khuyến nghị: Chainlit
chainlit run chatbot.py -w

# Hoặc: Gradio  
python app_gradio.py

# Hoặc: Streamlit
streamlit run app_modern.py
```

Chúc bạn sử dụng vui vẻ! 🎉
