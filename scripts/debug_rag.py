# scripts/debug_rag.py
import logging
import yaml
from pipeline.rag_pipeline import RAGPipeline
from retriever.faiss_retriever import FaissRetriever  # giả sử class retriever của bạn

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    # Load config
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Khởi tạo pipeline
    rag = RAGPipeline(cfg)

    # Khởi tạo retriever riêng để debug
    index_file = cfg['paths']['faiss_index']
    mapping_file = cfg['paths']['mapping_file']
    retriever = FaissRetriever(index_file=index_file, mapping_file=mapping_file, embedding_model=rag.retriever.model)

    # Một số câu hỏi test
    test_questions = [
        "How should crimes committed by a person be handled according to the law?"
    ]

    for i, q in enumerate(test_questions, 1):
        print(f"\n====== TEST QUESTION {i} ======")
        print(f"Question: {q}")

        # Encode query và search
        q_vec = retriever.encode_query(q)
        distances, indices = retriever.index.search(q_vec, 5)
        print("FAISS distances:", distances)
        print("FAISS indices:", indices)

        # Lấy top chunks
        retrieved_chunks = retriever.get_top_k(q, top_k=5)
        print(f"Number of retrieved chunks: {len(retrieved_chunks)}")
        if len(retrieved_chunks) == 0:
            print("⚠️ Warning: No chunks retrieved. Check if FAISS index and mapping match the embeddings.")

        for idx, chunk in enumerate(retrieved_chunks):
            # chunk là dict
            print(f"\n[Chunk {idx+1}]")
            print(f"Source: {chunk.get('source', 'N/A')}")
            print(f"Score: {chunk.get('distance', 'N/A')}")
            print(f"Text preview: {chunk.get('text', '')[:200]}...")

        # Sinh prompt cho LLM
        context_text = "\n".join([chunk.get("text", "") for chunk in retrieved_chunks])
        prompt = f"{cfg['prompt']['instruction']}:\n{context_text}\nQuestion: {q}\nAnswer:"
        logging.info(f"Generated prompt:\n{prompt[:1000]}...")  # in tối đa 1000 ký tự

        # Lấy kết quả LLM
        response = rag.llm.generate(prompt)
        print(f"\nLLM Output:\n{response}")

if __name__ == "__main__":
    main()
