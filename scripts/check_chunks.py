import os
import json

chunks_file = "data/processed/chunks.json"

if not os.path.exists(chunks_file):
    print(f"{chunks_file} not found!")
else:
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"{len(chunks)} chunks found.")
    # In 3 chunk đầu để kiểm tra
    for c in chunks[:3]:
        print("Chunk ID:", c.get("chunk_id"))
        print("Source:", c.get("source"))
        print("Text snippet:", c.get("chunk")[:100])
        print("-" * 50)
