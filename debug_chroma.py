from dotenv import load_dotenv
load_dotenv()

from src.vector_store import list_files, query_documents

def main():
    print("Files in Chroma:", list_files())
    res = query_documents("test", k=3)
    print("Query result keys:", res.keys())
    print("Docs len:", len(res["documents"][0]))

if __name__ == "__main__":
    main()
