import logging
import json
import os
from datetime import datetime

def setup_logger(log_file="logs/pipeline.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def log_query(query, chunks, answer, log_file="logs/query_logs.json"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "chunks": chunks,
        "answer": answer
    }
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def log_error(error_msg, error_file="logs/error_logs.txt"):
    os.makedirs(os.path.dirname(error_file), exist_ok=True)
    with open(error_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: {error_msg}\n")
    logging.error(error_msg)
