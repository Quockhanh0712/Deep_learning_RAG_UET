# 📋 Quick Reference - RAG System

## 🚀 Quick Start

```powershell
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Run app
streamlit run app.py

# 3. Upload PDF/TXT và hỏi đáp!
```

---

## ⚙️ Configuration (.env)

```bash
# LLM
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3

# Chunking
CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K=5

# Performance
EMBEDDING_BATCH_SIZE=32
USE_FP16=true
```

---

## 📊 Performance

- **Embedding**: 50-70 texts/s (GPU)
- **LLM**: 40-50 tokens/s
- **Total Query**: 1-2 seconds
- **VRAM**: 4-5GB / 6GB
- **Vietnamese Accuracy**: 85-90%

---

## 🔧 Common Commands

```powershell
# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Test GPU embeddings
python test_gpu_embedding.py

# Check Ollama
ollama list
ollama run qwen2.5:3b

# Monitor GPU
# Task Manager → Performance → GPU
```

---

## 🐛 Quick Fixes

### CUDA not available
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Ollama error
```powershell
ollama pull qwen2.5:3b
```

### Out of memory
```bash
# In .env
EMBEDDING_BATCH_SIZE=16
```

---

## 📁 Important Files

- `app.py` - Main UI
- `.env` - Configuration
- `src/embeddings.py` - BGE-M3 embeddings
- `src/llm_client.py` - Ollama/Gemini
- `src/rag_pipeline.py` - RAG logic

---

## 🎯 Key Components

1. **BGE-M3** - Multilingual embeddings (GPU)
2. **ChromaDB** - Vector store
3. **Qwen 2.5 3B** - Local LLM (via Ollama)
4. **Streamlit** - Web UI

---

**Status**: ✅ Production Ready  
**Version**: 2.0
