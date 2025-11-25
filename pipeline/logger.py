import logging
import json
import os
from datetime import datetime
import numpy as np

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

def make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def log_query(query, chunks, answer, log_file="logs/query_logs.json"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Convert chunks & answer to JSON serializable
    chunks = make_json_serializable(chunks)
    answer = make_json_serializable(answer)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "chunks": chunks,
        "answer": answer
    }
    
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
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
