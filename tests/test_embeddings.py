# tests/test_embeddings.py
from src.embeddings import embed_texts

def test_embed_texts_basic():
    texts = ["xin chào", "RAG là gì?"]
    vecs = embed_texts(texts)

    assert len(vecs) == len(texts)
    assert all(isinstance(v, list) for v in vecs)
    # vector không rỗng
    assert all(len(v) > 0 for v in vecs)
