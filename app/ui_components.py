import streamlit as st
import yaml

# --- Load config ---
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
CHUNK_DISPLAY_LIMIT = cfg.get("pipeline", {}).get("top_k_display", 5)
PREVIEW_HEIGHT = cfg.get("ui", {}).get("preview_height", 200)

# --- File uploader ---
def file_uploader_widget(label="Upload File", type_list=["pdf", "txt"]):
    uploaded_file = st.file_uploader(label, type=type_list)
    if uploaded_file:
        try:
            content = uploaded_file.read()
            if uploaded_file.type == "application/pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            else:
                text = content.decode("utf-8")
            return text, uploaded_file.name
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None, None
    return None, None

# --- File preview ---
def display_file_preview(text, height=PREVIEW_HEIGHT):
    st.text_area("File preview", text, height=height)

# --- Chat input ---
def chat_input_widget(label="Enter your question"):
    return st.text_input(label)

# --- Display answer ---
def display_answer(answer):
    st.subheader("Answer")
    st.success(answer)

# --- Display top-k chunks ---
def display_chunks(chunks, limit=CHUNK_DISPLAY_LIMIT):
    st.subheader(f"Top-{limit} Retrieved Chunks")
    for i, chunk in enumerate(chunks[:limit]):
        st.markdown(f"**Chunk {i+1}:** {chunk['text']}")
        metadata = chunk.get("metadata", {})
        if metadata:
            meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
            st.markdown(f"_Metadata_: {meta_str}")
        st.markdown("---")

# --- Utility: clear session ---
def clear_chat():
    for key in st.session_state.keys():
        del st.session_state[key]
