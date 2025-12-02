import uuid
from pathlib import Path
from typing import List
from pypdf import PdfReader
import logging
logger = logging.getLogger(__name__)
from src.vector_store import add_documents, delete_by_file_id

def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _chunk_text(content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = content.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max(1, chunk_size - chunk_overlap)
    return chunks

def load_file(path_str: str, chunk_size: int = 500, chunk_overlap: int = 100) -> str:
    path = Path(path_str)
    ext = path.suffix.lower()
    logger.info(f"[FILE_MANAGER] Loading file: {path} (exists={path.exists()})")
    
    # Read file content
    try:
        if ext == ".pdf":
            content = _read_pdf(path)
        else:
            content = _read_txt(path)
    except Exception as e:
        logger.error(f"[FILE_MANAGER] Error reading file: {e}")
        raise ValueError(f"Could not read file {path.name}: {str(e)}")

    logger.info(f"[FILE_MANAGER] Content length: {len(content)} chars")
    
    # Validate content is not empty
    if not content or len(content.strip()) < 10:
        error_msg = f"File {path.name} has no readable content or is too short (only {len(content)} chars)"
        logger.error(f"[FILE_MANAGER] {error_msg}")
        raise ValueError(error_msg)

    # Chunk the text
    chunks = _chunk_text(content, chunk_size, chunk_overlap)
    logger.info(f"[FILE_MANAGER] Num chunks: {len(chunks)}")
    
    # Validate chunks
    if not chunks:
        error_msg = f"No chunks created from file {path.name}"
        logger.error(f"[FILE_MANAGER] {error_msg}")
        raise ValueError(error_msg)

    # Create metadata
    file_id = f"{path.name}-{uuid.uuid4().hex[:8]}"
    ids = [f"{file_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"file_id": file_id, "chunk_index": i, "source": path.name}
        for i in range(len(chunks))
    ]
    
    # Add to vector store
    add_documents(chunks, metadatas, ids)
    logger.info(f"[FILE_MANAGER] Inserted {len(chunks)} chunks, file_id={file_id}")
    return file_id

def delete_file(file_id: str):
    delete_by_file_id(file_id)
