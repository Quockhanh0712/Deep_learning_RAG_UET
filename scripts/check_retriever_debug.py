import os
import pickle
import numpy as np
import faiss
import logging
from embeddings.embedding_model import EmbeddingModel
from pipeline.indexing import INDEX_CONFIG

logging.basicConfig(level=logging.INFO)

INDEX_PATH = "faiss_index/index.faiss"
MAPPING_PATH = "faiss_index/mapping.pkl"
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def load_index_and_mapping():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Mapping not found: {MAPPING_PATH}")

    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)

    return index, mapping


def check_vectors(mapping):
    """
    Kiểm tra embedding trong mapping
    """
    valid = 0
    norms = []
    dims = set()

    for i, item in enumerate(mapping):
        vec = item.get("embedding", None)

        if vec is None:
            print(f"❌ Chunk {i} không có embedding!")
            continue

        if not isinstance(vec, np.ndarray):
            print(f"❌ Chunk {i} embedding không phải numpy array!")
            continue

        dims.add(vec.shape[0])
        norm = np.linalg.norm(vec)
        norms.append(norm)

        valid += 1

    print("\n--- EMBEDDING CHECK ---")
    print(f"Tổng chunk: {len(mapping)}")
    print(f"Chunk có embedding hợp lệ: {valid}")
    print(f"Dimension các vector: {dims}")
    print(f"Norm min: {min(norms):.6f}")
    print(f"Norm max: {max(norms):.6f}")
    print(f"Norm mean: {np.mean(norms):.6f}")

    return norms


def check_index(index, mapping):
    """
    Kiểm tra FAISS index
    """
    print("\n--- FAISS INDEX CHECK ---")
    print("FAISS index dimension:", index.d)
    print("Vectors in index:", index.ntotal)
    print("Mapping size:", len(mapping))

    if index.ntotal != len(mapping):
        print("⚠️ SỐ VECTOR ≠ SỐ MAPPING → Có khả năng lệch index!")
    else:
        print("✅ Index và mapping đồng bộ.")


def test_raw_faiss_search(index, mapping):
    """
    Test search trực tiếp bằng chính vector đã lưu
    """
    print("\n--- TEST SEARCH WITH INTERNAL VECTOR ---")
    test_vec = mapping[0]["embedding"].astype("float32").reshape(1, -1)
    D, I = index.search(test_vec, k=5)

    print("Vector gốc search chính nó:")
    print("Indices:", I)
    print("Scores:", D)

    if I[0][0] != 0:
        print("❌ Vector 0 search lại không trả về chính nó -> sai index")
    else:
        print("✅ FAISS index hoạt động bình thường với vector gốc.")


def test_query(query, index, mapping, model):
    """
    Test query embedding + normalize + search
    """
    print(f"\n--- TEST QUERY: {query} ---")

    q_vec = model.encode([query])[0].astype("float32")
    norm_before = np.linalg.norm(q_vec)

    if INDEX_CONFIG["metric"].lower() == "cosine":
        faiss.normalize_L2(q_vec.reshape(1, -1))

    norm_after = np.linalg.norm(q_vec)

    print("Query norm trước normalize:", norm_before)
    print("Query norm sau normalize:", norm_after)

    D, I = index.search(q_vec.reshape(1, -1), 5)

    print("FAISS raw indices:", I)
    print("FAISS raw scores:", D)

    if np.all(I < 0):
        print("❌ FAISS không trả về kết quả nào!")
        return

    print("\n--- TOP 5 RESULTS ---")
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        chunk = mapping[idx]

        print(f"\n[{rank+1}] Index: {idx}")
        print("Score:", D[0][rank])
        print("Source:", chunk.get("source"))
        print("Text preview:", chunk["chunk"][:200].replace("\n", " "))


def main():
    print("\n====== RAG RETRIEVER FULL DEBUG ======")

    index, mapping = load_index_and_mapping()

    if isinstance(mapping, dict):
        print("⚠️ Mapping đang là dict, convert sang list...")
        mapping = [mapping[i] for i in sorted(mapping.keys())]

    norms = check_vectors(mapping)
    check_index(index, mapping)
    test_raw_faiss_search(index, mapping)

    print("\n--- LOAD EMBEDDING MODEL ---")
    model = EmbeddingModel(model_name=MODEL_NAME)

    test_query("How should crimes committed by a person be handled according to the law?", index, mapping, model)

if __name__ == "__main__":
    main()
