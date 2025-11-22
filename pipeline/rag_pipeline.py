import logging
import yaml
from embeddings.embedding_model import EmbeddingModel
from retriever.faiss_retriever import FaissRetriever
from llm.llm_wrapper import LLMWrapper
from llm.prompt_template import format_prompt
from pipeline.logger import log_query


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.top_k = cfg.get("indexing", {}).get("top_k", 5)

        # Load embedding model
        self.embedding_model = EmbeddingModel(
            model_name=cfg["embedding"]["model_name"]
        )

        # Load retriever
        self.retriever = FaissRetriever(
            index_file=cfg["paths"]["faiss_index"],
            mapping_file=cfg["paths"]["mapping_file"],
            embedding_model=self.embedding_model
        )

        # Load LLM
        self.llm = LLMWrapper()


        # Prompt instruction
        self.instruction = cfg.get("prompt", {}).get(
            "instruction",
            "Bạn là trợ lý AI. Trả lời dựa trên thông tin từ tài liệu."
        )

    def run(self, query: str):
        # Retrieve top-k chunks
        chunks = self.retriever.get_top_k(query, top_k=self.top_k)

        # Gom context từ các chunk
        context = "\n---\n".join([c["text"] for c in chunks])

        # Build prompt
        prompt = format_prompt(
            context=context,
            question=query,
            instruction=self.instruction
        )

        # Generate answer
        answer = self.llm.generate(prompt)

        # Log query
        log_query(query, chunks, answer)

        return answer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = load_config()
    rag = RAGPipeline(cfg)

    while True:
        query = input("Nhập câu hỏi: ")
        if query.lower() in ["exit", "quit"]:
            break

        answer = rag.run(query)
        print("Answer:", answer)
