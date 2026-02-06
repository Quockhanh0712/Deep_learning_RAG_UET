"""
Professional RAG Chatbot Interface using Chainlit
A modern, fast, and beautiful chatbot UI with Advanced RAG features:
- Hybrid Search (BM25 + Dense)
- Cross-Encoder Reranking
- Query Transformation
- Citation System
"""

import os
from pathlib import Path
from typing import List, Optional
import asyncio
from datetime import datetime

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.file_manager import load_file, delete_file
from src.vector_store import list_files, query_documents
from src.llm_client import generate_answer
from src.chat_history import get_history_manager

# Configuration
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = int(os.getenv("TOP_K", "5"))

# Advanced RAG settings
USE_ADVANCED_RAG = os.getenv("USE_ADVANCED_RAG", "true").lower() == "true"
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
USE_CITATIONS = os.getenv("USE_CITATIONS", "true").lower() == "true"


def build_prompt(question: str, docs: List[str]) -> str:
    """Build the prompt for the LLM."""
    context = "\n\n".join(docs)
    prompt = (
        "Bạn là trợ lý AI thông minh và thân thiện. "
        "Trả lời chi tiết, chính xác và dễ hiểu dựa trên ngữ cảnh được cung cấp.\n"
        "Nếu không tìm thấy thông tin phù hợp, hãy thông báo rằng bạn không có đủ thông tin.\n\n"
        f" Ngữ cảnh:\n{context}\n\n"
        f"❓ Câu hỏi: {question}\n\n"
        "💡 Trả lời:"
    )
    return prompt


def sync_process_query_basic(question: str) -> dict:
    """Basic RAG query (legacy mode)."""
    res = query_documents(question, k=TOP_K)
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    
    prompt = build_prompt(question, docs)
    answer = generate_answer(prompt)
    
    return {
        "answer": answer,
        "context": docs,
        "metadatas": metadatas,
        "citations_md": "",
    }


def sync_process_query_advanced(question: str) -> dict:
    """Advanced RAG query with hybrid search, reranking, and citations."""
    try:
        from src.retrieval import advanced_query
        
        result = advanced_query(
            question=question,
            use_hybrid=USE_HYBRID_SEARCH,
            use_rerank=USE_RERANKER,
            use_citations=USE_CITATIONS,
            top_k=TOP_K
        )
        
        # Extract context and metadata from sources
        docs = [s.get("content", "") for s in result.sources]
        metadatas = [s.get("metadata", {}) for s in result.sources]
        
        # Add retrieval info to answer if available
        answer = result.answer
        
        # Add query transformation info
        if result.query_info.get("rewritten") and result.query_info["rewritten"] != result.query_info["original"]:
            answer += f"\n\n_🔄 Query đã được tối ưu: \"{result.query_info['rewritten']}\"_"
        
        return {
            "answer": answer,
            "context": docs,
            "metadatas": metadatas,
            "citations_md": result.citations_markdown,
            "retrieval_info": result.retrieval_info,
        }
        
    except ImportError as e:
        # Fallback to basic if advanced modules not available
        print(f"[CHATBOT] Advanced RAG not available, falling back to basic: {e}")
        return sync_process_query_basic(question)
    except Exception as e:
        print(f"[CHATBOT] Advanced RAG error, falling back to basic: {e}")
        return sync_process_query_basic(question)


def sync_process_query(question: str) -> dict:
    """Process a question and return the answer with context."""
    if USE_ADVANCED_RAG:
        return sync_process_query_advanced(question)
    else:
        return sync_process_query_basic(question)


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
    # Generate session ID
    session_id = cl.user_session.get("id")
    
    # Initialize chat history
    history_mgr = get_history_manager()
    conv_id = history_mgr.get_or_create_conversation(session_id)
    
    cl.user_session.set("conversation_id", conv_id)
    cl.user_session.set("history", [])
    
    # Load existing messages from database
    messages = history_mgr.get_conversation_messages(conv_id)
    if messages:
        loaded_history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        cl.user_session.set("history", loaded_history)
        
        # Notify user about loaded history
        await cl.Message(
            content=f"📜 Đã tải {len(messages)} tin nhắn từ lịch sử."
        ).send()
    
    # Get list of loaded files
    files = list_files()
    
    # Build feature status
    features = []
    if USE_ADVANCED_RAG:
        features.append("🚀 Advanced RAG")
    if USE_HYBRID_SEARCH:
        features.append("🔍 Hybrid Search")
    if USE_RERANKER:
        features.append("📊 Reranking")
    if USE_CITATIONS:
        features.append("📚 Citations")
    
    feature_str = " | ".join(features) if features else "Basic RAG"
    
    # Welcome message
    welcome_msg = f"""# 🤖 RAG Chatbot Pro

Xin chào! Tôi là trợ lý AI.

**✨ Tính năng:** {feature_str}

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

**📱 Quick Commands:**
- `/history` - 📜 Xem lịch sử conversations
- `/search <query>` - 🔍 Tìm kiếm trong lịch sử
- `/export` - 📥 Export conversation hiện tại
- `/stats` - 📊 Xem thống kê hệ thống
- `/files` - 📁 Danh sách tài liệu
- `/clear` - 🧹 Xóa lịch sử chat
- `/help` - ❓ Hiển thị trợ giúp

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
- `/mode` - Xem chế độ RAG hiện tại
- `/clear` - Xóa lịch sử chat
- `/help` - Hiển thị trợ giúp

**Upload file:**
- Click vào icon 📎 để đính kèm file PDF/TXT

**Hỏi đáp:**
- Nhập câu hỏi liên quan đến tài liệu đã nạp

**Advanced Features:**
- 🔍 Hybrid Search: Kết hợp BM25 + Dense search
- 📊 Reranking: Cross-encoder để xếp hạng lại kết quả
- 📚 Citations: Hiển thị nguồn tham khảo
"""
        await cl.Message(content=help_msg).send()
        return
    
    if user_input.lower() == "/mode":
        mode_msg = f"""## ⚙️ Chế độ RAG

**Advanced RAG:** {'✅ Bật' if USE_ADVANCED_RAG else '❌ Tắt'}
**Hybrid Search:** {'✅ Bật' if USE_HYBRID_SEARCH else '❌ Tắt'}
**Reranking:** {'✅ Bật' if USE_RERANKER else '❌ Tắt'}
**Citations:** {'✅ Bật' if USE_CITATIONS else '❌ Tắt'}

_Để thay đổi, cập nhật biến môi trường và khởi động lại._

**Biến môi trường:**
- `USE_ADVANCED_RAG=true/false`
- `USE_HYBRID_SEARCH=true/false`
- `USE_RERANKER=true/false`
- `USE_CITATIONS=true/false`
"""
        await cl.Message(content=mode_msg).send()
        return
    
    if user_input.lower().startswith("/delete "):
        file_id = user_input[8:].strip()
        try:
            delete_file(file_id)
            await cl.Message(content=f"🗑️ Đã xóa file: `{file_id}`").send()
        except Exception as e:
            await cl.Message(content=f"❌ Lỗi khi xóa file: {str(e)}").send()
        return
    
    # NEW: Chat history commands
    if user_input.lower() == "/history":
        """Show conversation history"""
        history_mgr = get_history_manager()
        conversations = history_mgr.list_conversations(limit=10)
        
        if conversations:
            msg = "# 📜 Lịch sử Conversations\n\n"
            current_conv_id = cl.user_session.get("conversation_id")
            
            for i, conv in enumerate(conversations, 1):
                is_current = "🟢 **ĐANG MỞ**" if conv.id == current_conv_id else ""
                msg += f"### {i}. {conv.title} {is_current}\n\n"
                msg += f"| | |\n|---|---|\n"
                msg += f"| **ID** | `{conv.id}` |\n"
                msg += f"| **Session** | `{conv.session_id[:12]}...` |\n"
                msg += f"| **Created** | {conv.created_at} |\n"
                msg += f"| **Updated** | {conv.updated_at} |\n\n"
                msg += "---\n\n"
            
            msg += f"\n_Hiển thị {len(conversations)} conversations gần nhất_"
            await cl.Message(content=msg).send()
        else:
            await cl.Message(content="📭 Chưa có lịch sử conversation.").send()
        return
    
    if user_input.lower().startswith("/search "):
        """Search in chat history"""
        query = user_input[8:].strip()
        if not query:
            await cl.Message(content="⚠️ Vui lòng nhập từ khóa tìm kiếm: `/search <từ khóa>`").send()
            return
        
        history_mgr = get_history_manager()
        results = history_mgr.search_messages(query, limit=5)
        
        if results:
            msg = f"# 🔍 Kết quả Tìm kiếm\n\n"
            msg += f"**Từ khóa:** `{query}`  \n"
            msg += f"**Số kết quả:** {len(results)}\n\n"
            msg += "---\n\n"
            
            for i, r in enumerate(results, 1):
                role_icon = "👤" if r['role'] == "user" else "🤖"
                content_preview = r['content'][:200] + "..." if len(r['content']) > 200 else r['content']
                
                msg += f"### {i}. {role_icon} {r['role'].title()}\n\n"
                msg += f"**From:** {r['conversation_title']}\n\n"
                msg += f"> {content_preview}\n\n"
                msg += f"_📅 {r['created_at']}_\n\n"
                msg += "---\n\n"
            
            await cl.Message(content=msg).send()
        else:
            await cl.Message(content=f"❌ Không tìm thấy kết quả cho `{query}`\n\n_Thử từ khóa khác hoặc kiểm tra chính tả_").send()
        return
    
    if user_input.lower() == "/export":
        """Export current conversation"""
        history_mgr = get_history_manager()
        conv_id = cl.user_session.get("conversation_id")
        
        try:
            md_content = history_mgr.export_conversation_markdown(conv_id)
            
            # Save to file
            export_dir = Path("data/exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"conversation_{conv_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            export_path.write_text(md_content, encoding="utf-8")
            
            await cl.Message(
                content=f"✅ Đã export conversation!\n\n📁 File: `{export_path}`\n\n_Download file từ folder `data/exports`_"
            ).send()
        except Exception as e:
            await cl.Message(content=f"❌ Lỗi khi export: {str(e)}").send()
        return
    
    if user_input.lower() == "/stats":
        """Show usage statistics"""
        history_mgr = get_history_manager()
        stats = history_mgr.get_stats()
        
        msg = f"""## 📊 Thống kê Hệ thống

**Tổng Conversations:** {stats['total_conversations']}
**Tổng Messages:** {stats['total_messages']}
**Trung bình Messages/Conversation:** {stats['avg_messages_per_conv']:.1f}

**Phân bố Messages:**
"""
        for role, count in stats.get('message_distribution', {}).items():
            msg += f"- {role.title()}: {count}\n"
        
        await cl.Message(content=msg).send()
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
        citations_md = result.get("citations_md", "")
        retrieval_info = result.get("retrieval_info", {})
        
        # Build full response with citations
        full_answer = answer
        if citations_md:
            full_answer += citations_md
        
        # Add retrieval info badge
        if retrieval_info:
            method = retrieval_info.get("method", "dense")
            reranked = retrieval_info.get("reranked", False)
            badges = []
            if method == "hybrid":
                badges.append("🔍 Hybrid")
            if reranked:
                badges.append("📊 Reranked")
            if badges:
                full_answer += f"\n\n_{'  '.join(badges)}_"
        
        # Update message with answer
        msg.content = full_answer
        
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
        
        # Save to database (persistent storage)
        history_mgr = get_history_manager()
        conv_id = cl.user_session.get("conversation_id")
        
        # Save user message
        history_mgr.add_message(
            conversation_id=conv_id,
            role="user",
            content=user_input
        )
        
        # Save assistant message with full context
        history_mgr.add_message(
            conversation_id=conv_id,
            role="assistant",
            content=answer,
            context=context,
            metadatas=metadatas,
            citations=citations_md
        )
        
        # Update in-memory history (for current session)
        history = cl.user_session.get("history", [])
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        cl.user_session.set("history", history)
        
    except Exception as e:
        msg.content = f"❌ Đã xảy ra lỗi: {str(e)}"
        await msg.update()
