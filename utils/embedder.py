import os, streamlit as st
from sentence_transformers import SentenceTransformer

HF_TOKEN = st.secrets["huggingface"]["token"]

@st.cache_resource
def get_embedder(
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
):
    return SentenceTransformer(model_name, token=HF_TOKEN)


def chunk_text(pages, chunk_size: int = 150):
    chunks = []
    for page in pages:
        words = page["text"].split()
        for i in range(0, len(words), chunk_size):
            snippet = " ".join(words[i : i + chunk_size]).strip()
            if snippet:
                chunks.append({"page_num": page["page_num"], "chunk": snippet})
    return chunks


def embed_chunks(chunks, model):
    texts = [c["chunk"] for c in chunks]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
