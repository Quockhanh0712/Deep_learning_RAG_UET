import streamlit as st

def show_context(context_docs, metadatas):
    for doc, meta in zip(context_docs, metadatas):
        st.markdown(
            f"**Source:** {meta.get('source')} (chunk {meta.get('chunk_index')})"
        )
        st.write(doc)
        st.markdown("---")
