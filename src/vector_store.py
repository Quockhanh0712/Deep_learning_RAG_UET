import os
import chromadb
from chromadb.config import Settings
from src.embeddings import embed_texts  # <-- thêm src.
from src.embeddings import ChromaEmbeddingFunction

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False),
)

_embedding_fn = ChromaEmbeddingFunction()

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_fn,  # dùng object này
)


def add_documents(doc_texts, metadatas, ids):
    _collection.add(documents=doc_texts, metadatas=metadatas, ids=ids)

def query_documents(query, k=5):
    return _collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )

def delete_by_file_id(file_id: str):
    all_docs = _collection.get(include=["metadatas"])
    ids_to_delete = [
        id_
        for id_, meta in zip(all_docs["ids"], all_docs["metadatas"])
        if meta.get("file_id") == file_id
    ]
    if ids_to_delete:
        _collection.delete(ids=ids_to_delete)

def list_files():
    all_docs = _collection.get(include=["metadatas"])
    files = set()
    for meta in all_docs["metadatas"]:
        fid = meta.get("file_id")
        if fid:
            files.add(fid)
    return sorted(files)
