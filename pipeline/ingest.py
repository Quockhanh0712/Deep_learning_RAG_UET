import os
import yaml
import logging
from pathlib import Path
import pdfplumber
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def parse_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n")

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_file(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".html":
        return parse_html(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        logging.warning(f"Unsupported file type: {file_path}")
        return ""

def save_raw_text(text, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def ingest_folder(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        text = load_file(file_path)
        if text:
            output_file = os.path.join(output_folder, f"{Path(file_name).stem}.txt")
            save_raw_text(text, output_file)
            logging.info(f"Ingested {file_name} → {output_file}")

if __name__ == "__main__":
    cfg = load_config()
    raw_folder = cfg.get("paths", {}).get("raw_data", "data/raw")
    processed_folder = cfg.get("paths", {}).get("processed_data", "data/processed")
    ingest_folder(raw_folder, processed_folder)
