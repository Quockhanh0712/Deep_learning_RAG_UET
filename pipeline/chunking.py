# pipeline/chunking.py
import os
import yaml
import json

# PDF extraction
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# Try to import RecursiveCharacterTextSplitter from langchain; fallback lightweight version
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    class RecursiveCharacterTextSplitter:
        """Lightweight fallback implementation"""
        def __init__(self, chunk_size=500, chunk_overlap=50):
            self.chunk_size = int(chunk_size)
            self.chunk_overlap = int(chunk_overlap)

        def split_text(self, text):
            if not text:
                return []
            size = self.chunk_size
            overlap = self.chunk_overlap
            chunks = []
            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + size, text_len)
                chunks.append(text[start:end])
                if end >= text_len:
                    break
                start = max(0, end - overlap)
                if start == end:
                    start = end
            return chunks

# --- Load chunking config ---
def load_chunking_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        return {"chunk_size": 500, "chunk_overlap": 50}  # default
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    chunk_cfg = cfg.get("chunking", {})
    return {
        "chunk_size": chunk_cfg.get("chunk_size", 500),
        "chunk_overlap": chunk_cfg.get("chunk_overlap", 50)
    }

CHUNK_CONFIG = load_chunking_config()

# --- Chunking functions ---
def chunk_text(text: str, source: str = "unknown") -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CONFIG["chunk_size"],
        chunk_overlap=CHUNK_CONFIG["chunk_overlap"]
    )
    texts = splitter.split_text(text)
    chunks = []
    for i, t in enumerate(texts):
        chunks.append({
            "chunk": t,
            "source": source,
            "chunk_id": i
        })
    return chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    if fitz is None:
        raise ImportError("PyMuPDF is not installed. Install with `pip install PyMuPDF`")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_file(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    # Optional clean
    text = text.replace("\n", " ").strip()
    source_name = os.path.basename(file_path)
    return chunk_text(text, source=source_name)

def chunk_documents(doc_texts: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in doc_texts:
        text = doc.get("text", "")
        source = doc.get("source", "unknown")
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
    return all_chunks

# --- Run as script ---
if __name__ == "__main__":
    input_dir = "data/raw"  # default raw files folder
    output_file = "data/processed/chunks.json"
    all_chunks = []

    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".txt", ".pdf")):
            file_path = os.path.join(input_dir, fname)
            chunks = process_file(file_path)
            all_chunks.extend(chunks)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {output_file}")
