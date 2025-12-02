"""
Professional RAG Chatbot Interface using Chainlit
A modern, fast, and beautiful chatbot UI for RAG-based Q&A
"""

import os
from pathlib import Path
from typing import List, Optional
import asyncio

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.file_manager import load_file, delete_file
from src.vector_store import list_files, query_documents
from src.llm_client import generate_answer

# Configuration
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = int(os.getenv("TOP_K", "5"))


def build_prompt(question: str, docs: List[str]) -> str:
    """Build the prompt for the LLM."""
    context = "\n\n".join(docs)
    prompt = (
        "Bạn là trợ lý AI thông minh và thân thiện. "
        "Trả lời chi tiết, chính xác và dễ hiểu dựa trên ngữ cảnh được cung cấp.\n"
        "Nếu không tìm thấy thông tin phù hợp, hãy thông báo rằng bạn không có đủ thông tin.\n\n"
        f"📚 Ngữ cảnh:\n{context}\n\n"
        f"❓ Câu hỏi: {question}\n\n"
        "💡 Trả lời:"
    )
    return prompt


def sync_process_query(question: str) -> dict:
    """Process a question and return the answer with context."""
    # Query documents
    res = query_documents(question, k=TOP_K)
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    
    # Build prompt and generate answer
    prompt = build_prompt(question, docs)
    answer = generate_answer(prompt)
    
    return {
        "answer": answer,
        "context": docs,
        "metadatas": metadatas,
    }


async def process_uploaded_file(file_path: str, file_name: str) -> str:
    """Process an uploaded file and add to vector store."""
    # Save to uploads directory
    save_path = UPLOAD_DIR / file_name
    
    # Copy file
    with open(file_path, "rb") as src:
        content = src.read()
    with open(save_path, "wb") as dst:
        dst.write(content)
    
    # Check if already loaded
    existing_files = list_files()
    already_loaded = any(fid.startswith(file_name + "-") for fid in existing_files)
    
    if already_loaded:
        return f"ℹ️ File `{file_name}` đã được nạp trước đó."
    
    # Load file into vector store
    file_id = await asyncio.to_thread(load_file, str(save_path))
    return f"✅ Đã nạp file thành công!\n- **File:** `{file_name}`\n- **ID:** `{file_id}`"


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Set session variables
    cl.user_session.set("history", [])
    
    # Get list of loaded files
    files = list_files()
    
    # Welcome message
    welcome_msg = """# 🤖 RAG Chatbot

Xin chào! Tôi là trợ lý AI được hỗ trợ bởi **BGE Embeddings** + **ChromaDB** + **Gemini/Ollama**.

## 📋 Tài liệu đã nạp:
"""
    
    if files:
        for f in files:
            welcome_msg += f"- 📄 `{f}`\n"
    else:
        welcome_msg += "_Chưa có tài liệu nào được nạp._\n"
    
    welcome_msg += """
---
**💡 Hướng dẫn:**
- Đặt câu hỏi về nội dung tài liệu đã nạp
- Đính kèm file PDF/TXT cùng với tin nhắn để upload
- Nhập `/files` để xem danh sách tài liệu
- Nhập `/clear` để xóa lịch sử chat
- Nhập `/delete <file_id>` để xóa tài liệu

Hãy đặt câu hỏi của bạn! 🚀
"""
    
    await cl.Message(content=welcome_msg).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    user_input = message.content.strip()
    
    # Handle file attachments first
    if message.elements:
        for element in message.elements:
            if hasattr(element, 'path') and element.path:
                file_name = getattr(element, 'name', os.path.basename(element.path))
                # Check file extension
                if file_name.lower().endswith(('.pdf', '.txt')):
                    await cl.Message(content=f"⏳ Đang xử lý file `{file_name}`...").send()
                    try:
                        result = await process_uploaded_file(element.path, file_name)
                        await cl.Message(content=result).send()
                    except Exception as e:
                        await cl.Message(content=f"❌ Lỗi khi nạp file: {str(e)}").send()
                else:
                    await cl.Message(content=f"⚠️ Chỉ hỗ trợ file PDF và TXT. File `{file_name}` bị bỏ qua.").send()
        
        # If only file upload, no question
        if not user_input:
            return
    
    # Handle special commands
    if user_input.lower() == "/files":
        files = list_files()
        if files:
            file_list = "\n".join([f"📄 `{f}`" for f in files])
            await cl.Message(content=f"## 📚 Tài liệu đã nạp:\n{file_list}").send()
        else:
            await cl.Message(content="📭 Chưa có tài liệu nào được nạp.").send()
        return
    
    if user_input.lower() == "/clear":
        cl.user_session.set("history", [])
        await cl.Message(content="🧹 Đã xóa lịch sử chat!").send()
        return
    
    if user_input.lower() == "/help":
        help_msg = """## 📖 Trợ giúp

**Commands:**
- `/files` - Xem danh sách tài liệu đã nạp
- `/delete <file_id>` - Xóa tài liệu
- `/clear` - Xóa lịch sử chat
- `/help` - Hiển thị trợ giúp

**Upload file:**
- Click vào icon 📎 để đính kèm file PDF/TXT

**Hỏi đáp:**
- Nhập câu hỏi liên quan đến tài liệu đã nạp
"""
        await cl.Message(content=help_msg).send()
        return
    
    if user_input.lower().startswith("/delete "):
        file_id = user_input[8:].strip()
        try:
            delete_file(file_id)
            await cl.Message(content=f"🗑️ Đã xóa file: `{file_id}`").send()
        except Exception as e:
            await cl.Message(content=f"❌ Lỗi khi xóa file: {str(e)}").send()
        return
    
    # Ignore empty messages
    if not user_input:
        return
    
    # Check if there are any documents
    files = list_files()
    if not files:
        await cl.Message(
            content="⚠️ Chưa có tài liệu nào được nạp. Vui lòng đính kèm file PDF/TXT để bắt đầu."
        ).send()
        return
    
    # Create response message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Get answer
        result = await asyncio.to_thread(sync_process_query, user_input)
        
        answer = result["answer"]
        context = result["context"]
        metadatas = result["metadatas"]
        
        # Update message with answer
        msg.content = answer
        
        # Show context as expandable elements
        if context:
            elements = []
            for i, (doc, meta) in enumerate(zip(context, metadatas)):
                source = meta.get("source", "Unknown")
                chunk_idx = meta.get("chunk_index", i)
                
                # Create text element for context
                elements.append(
                    cl.Text(
                        name=f"📖 Nguồn {i+1}: {source} (chunk {chunk_idx})",
                        content=doc,
                        display="side"
                    )
                )
            
            msg.elements = elements
        
        await msg.update()
        
        # Update history
        history = cl.user_session.get("history", [])
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        cl.user_session.set("history", history)
        
    except Exception as e:
        msg.content = f"❌ Đã xảy ra lỗi: {str(e)}"
        await msg.update()
