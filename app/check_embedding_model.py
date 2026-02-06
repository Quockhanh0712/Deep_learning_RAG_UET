"""
Check which embedding model is currently loaded and GPU usage
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("EMBEDDING MODEL CHECK")
print("=" * 60)

# Check environment
import os
model_name = os.getenv("EMBEDDING_MODEL", "N/A")
device = os.getenv("EMBEDDING_DEVICE", "cpu")
print(f"\n📝 Environment Config:")
print(f"   EMBEDDING_MODEL: {model_name}")
print(f"   EMBEDDING_DEVICE: {device}")

# Check GPU
import torch
print(f"\n🖥️ PyTorch CUDA:")
print(f"   Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Try loading the pipeline
print(f"\n⏳ Loading Legal RAG Pipeline...")
try:
    from src.legal_rag_pipeline import get_legal_rag_pipeline
    pipeline = get_legal_rag_pipeline()
    pipeline.initialize()
    
    print(f"\n✅ Pipeline loaded successfully!")
    
    # Check embedding model
    if hasattr(pipeline, 'embedding_model'):
        emb = pipeline.embedding_model
        print(f"\n📊 Embedding Model Info:")
        print(f"   Model Name: {emb.model_name}")
        print(f"   Device: {emb.device}")
        
        if torch.cuda.is_available():
            print(f"\n🔥 GPU Memory After Loading:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Test encoding
        print(f"\n🧪 Testing encoding...")
        test_text = "Điều 123 của Bộ luật Hình sự"
        embeddings = emb.encode([test_text])
        print(f"   Input: '{test_text}'")
        print(f"   Output shape: {len(embeddings)}x{len(embeddings[0])}")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        
    # Check stats
    stats = pipeline.get_stats()
    print(f"\n📈 Pipeline Stats:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
except Exception as e:
    import traceback
    print(f"\n❌ Error loading pipeline:")
    traceback.print_exc()

print("\n" + "=" * 60)
