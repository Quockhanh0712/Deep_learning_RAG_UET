# test_gpu_embedding.py
import torch
from src.embeddings import get_model, embed_texts
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
model = get_model()
print(f"Model device: {model.device}")
# Benchmark
import time
texts = ["Test sentence"] * 100
start = time.time()
embeddings = embed_texts(texts, batch_size=32)
end = time.time()
print(f"Embedded {len(texts)} texts in {end-start:.2f}s")
print(f"Speed: {len(texts)/(end-start):.1f} it/s")