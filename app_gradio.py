"""
Professional RAG Chatbot Interface using Gradio
Fast, beautiful, and feature-rich chat interface
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import gradio as gr
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from src.file_manager import load_file, delete_file
from src.vector_store import list_files, query_documents
from src.llm_client import generate_answer

# Constants
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = int(os.getenv("TOP_K", "5"))


def build_prompt(question: str, docs: List[str]) -> str:
    """Build the prompt for the LLM."""
    context = "\n\n".join(docs)
    prompt = (
        "Bạn là trợ lý AI thông minh và thân thiện. "
        "Trả lời chi tiết, chính xác và dễ hiểu dựa trên ngữ cảnh được cung cấp.\n"
        "Sử dụng markdown để format câu trả lời đẹp hơn khi cần thiết.\n"
        "Nếu không tìm thấy thông tin phù hợp, hãy thông báo rằng bạn không có đủ thông tin.\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời:"
    )
    return prompt


def process_query(question: str) -> dict:
    """Process a question and return the answer with context."""
    res = query_documents(question, k=TOP_K)
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    
    prompt = build_prompt(question, docs)
    answer = generate_answer(prompt)
    
    return {
        "answer": answer,
        "context": docs,
        "metadatas": metadatas,
    }


def get_files_list() -> str:
    """Get formatted list of loaded files."""
    files = list_files()
    if not files:
        return "📭 Chưa có tài liệu nào được nạp."
    
    file_list = "\n".join([f"📄 {f}" for f in files])
    return f"**📚 Tài liệu đã nạp ({len(files)}):**\n\n{file_list}"


def upload_file(file) -> str:
    """Handle file upload."""
    if file is None:
        return "⚠️ Vui lòng chọn file để tải lên."
    
    try:
        file_name = os.path.basename(file.name)
        save_path = UPLOAD_DIR / file_name
        
        # Copy file to upload directory
        with open(file.name, "rb") as src:
            content = src.read()
        with open(save_path, "wb") as dst:
            dst.write(content)
        
        # Check if already loaded
        existing_files = list_files()
        already_loaded = any(fid.startswith(file_name + "-") for fid in existing_files)
        
        if already_loaded:
            return f"ℹ️ File `{file_name}` đã được nạp trước đó."
        
        # Load file
        file_id = load_file(str(save_path))
        return f"✅ Đã nạp file thành công!\n\n**File:** `{file_name}`\n**ID:** `{file_id}`"
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return f"❌ Lỗi khi tải file: {str(e)}"


def delete_document(file_id: str) -> str:
    """Delete a document from the vector store."""
    if not file_id.strip():
        return "⚠️ Vui lòng nhập ID file cần xóa."
    
    try:
        delete_file(file_id.strip())
        return f"🗑️ Đã xóa file: `{file_id}`"
    except Exception as e:
        return f"❌ Lỗi khi xóa file: {str(e)}"


def chat(
    message: str,
    history: List[Tuple[str, str]]
) -> Tuple[str, List[Tuple[str, str]], str]:
    """Handle chat messages."""
    
    if not message.strip():
        return "", history, ""
    
    # Check if there are documents
    files = list_files()
    if not files:
        response = "⚠️ Chưa có tài liệu nào được nạp. Vui lòng tải lên file trước khi đặt câu hỏi."
        history.append((message, response))
        return "", history, ""
    
    try:
        # Process query
        result = process_query(message)
        answer = result["answer"]
        context = result["context"]
        metadatas = result["metadatas"]
        
        # Format context for display
        context_text = "### 📚 Nguồn tham khảo:\n\n"
        for i, (doc, meta) in enumerate(zip(context, metadatas)):
            source = meta.get("source", "Unknown")
            chunk_idx = meta.get("chunk_index", i)
            preview = doc[:300] + "..." if len(doc) > 300 else doc
            context_text += f"**{i+1}. {source}** (chunk {chunk_idx})\n> {preview}\n\n---\n\n"
        
        history.append((message, answer))
        return "", history, context_text
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_msg = f"❌ Đã xảy ra lỗi: {str(e)}"
        history.append((message, error_msg))
        return "", history, ""


def clear_chat() -> Tuple[List, str]:
    """Clear chat history."""
    return [], ""


# Custom CSS
custom_css = """
/* Global Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
}

.header-container h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.header-container p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
    font-size: 1.1rem;
}

/* Chat area */
.chatbot {
    border-radius: 16px !important;
    border: 1px solid #e9ecef !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
}

/* Message bubbles */
.message {
    border-radius: 16px !important;
    padding: 1rem !important;
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.bot-message {
    background: #f8f9fa !important;
}

/* Input area */
.input-row {
    gap: 0.5rem;
}

#chat-input {
    border-radius: 25px !important;
    border: 2px solid #e9ecef !important;
    padding: 1rem 1.5rem !important;
    font-size: 1rem !important;
}

#chat-input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

.secondary-btn {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    border-radius: 25px !important;
    color: #666 !important;
}

/* Sidebar */
.sidebar-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.sidebar-section h3 {
    margin-top: 0;
    color: #333;
}

/* Context panel */
.context-panel {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    max-height: 400px;
    overflow-y: auto;
}

/* Tabs */
.tab-nav button {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* File upload */
.file-upload {
    border: 2px dashed #667eea !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    text-align: center;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: fadeIn 0.3s ease-out;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}
"""

# Build the interface
with gr.Blocks() as demo:
    
    # Apply custom CSS
    gr.HTML(f"<style>{custom_css}</style>")
    
    # Header
    gr.HTML("""
    <div class="header-container">
        <h1>🤖 RAG Chatbot</h1>
        <p>Trợ lý AI thông minh - Hỏi đáp trên tài liệu của bạn</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            Powered by BGE-M3 + ChromaDB + Gemini
        </p>
    </div>
    """)
    
    with gr.Row():
        # Main chat area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Cuộc trò chuyện",
                height=500,
                show_label=True,
                container=True,
                bubble_full_width=False,
                avatar_images=(None, "🤖"),
                elem_classes=["chatbot"]
            )
            
            with gr.Row(elem_classes=["input-row"]):
                msg_input = gr.Textbox(
                    placeholder="💬 Nhập câu hỏi của bạn...",
                    show_label=False,
                    container=False,
                    scale=6,
                    elem_id="chat-input"
                )
                
                send_btn = gr.Button(
                    "📤 Gửi",
                    variant="primary",
                    scale=1,
                    elem_classes=["primary-btn"]
                )
                
                clear_btn = gr.Button(
                    "🧹 Xóa",
                    variant="secondary", 
                    scale=1,
                    elem_classes=["secondary-btn"]
                )
        
        # Sidebar
        with gr.Column(scale=1):
            with gr.Tabs():
                # Documents tab
                with gr.TabItem("📚 Tài liệu", id="docs"):
                    files_display = gr.Markdown(
                        value=get_files_list,
                        every=5  # Refresh every 5 seconds
                    )
                    
                    refresh_btn = gr.Button(
                        "🔄 Làm mới danh sách",
                        size="sm"
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 📤 Tải lên tài liệu")
                    
                    file_upload = gr.File(
                        label="Chọn file (PDF/TXT)",
                        file_types=[".pdf", ".txt"],
                        type="filepath"
                    )
                    
                    upload_status = gr.Markdown("")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🗑️ Xóa tài liệu")
                    
                    delete_input = gr.Textbox(
                        label="ID file cần xóa",
                        placeholder="Nhập file ID..."
                    )
                    
                    delete_btn = gr.Button(
                        "🗑️ Xóa",
                        variant="stop",
                        size="sm"
                    )
                    
                    delete_status = gr.Markdown("")
                
                # Context tab
                with gr.TabItem("📖 Nguồn", id="context"):
                    context_display = gr.Markdown(
                        value="*Nguồn tham khảo sẽ hiển thị ở đây sau khi bạn đặt câu hỏi.*",
                        elem_classes=["context-panel"]
                    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 1rem; color: #888; font-size: 0.85rem;">
        <p>🔧 Được xây dựng với BGE-M3 Embeddings + ChromaDB + Google Gemini</p>
        <p>Made with ❤️ for RAG applications</p>
    </div>
    """)
    
    # Event handlers
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, context_display]
    )
    
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, context_display]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, context_display]
    )
    
    refresh_btn.click(
        fn=get_files_list,
        outputs=[files_display]
    )
    
    file_upload.change(
        fn=upload_file,
        inputs=[file_upload],
        outputs=[upload_status]
    ).then(
        fn=get_files_list,
        outputs=[files_display]
    )
    
    delete_btn.click(
        fn=delete_document,
        inputs=[delete_input],
        outputs=[delete_status]
    ).then(
        fn=get_files_list,
        outputs=[files_display]
    )


# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None
    )
