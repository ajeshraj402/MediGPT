import os
import json
import faiss
import streamlit as st
from tempfile import NamedTemporaryFile

# Streamlit command
st.set_page_config(page_title="MediPDF Chatbot", page_icon="🩺")

# project imports
from utils.pdf_parser  import extract_text_by_page
from utils.embedder    import chunk_text, get_embedder, embed_chunks
from utils.retriever   import load_faiss_and_chunks, retrieve_top_k_chunks
from utils.rag_engine  import get_flan_pipeline, build_prompt

# load cached models once
qa_pipeline  = get_flan_pipeline()
model_embed  = get_embedder()

# Streamlit UI
st.title("🩺 Medical PDF Chatbot with Page Sources")
st.markdown(
    "Upload a **medical PDF** and ask questions about its content. "
    "Answers cite the pages they came from."
)

uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded successfully!")

    # extract, chunk, embed
    pages      = extract_text_by_page(pdf_path)
    chunks     = chunk_text(pages, chunk_size=150)
    embeddings = embed_chunks(chunks, model_embed)

    # build in-memory FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # keep in session_state
    st.session_state.chunks = chunks
    st.session_state.index  = index

if "chunks" in st.session_state and "index" in st.session_state:
    question = st.text_input("🧠 Ask a question about the PDF:")

    if question:
        top_chunks = retrieve_top_k_chunks(
            question,
            model_embed,
            st.session_state.index,
            st.session_state.chunks,
            k=3,
        )

        prompt, _ = build_prompt(question, top_chunks)
        response  = qa_pipeline(prompt, max_new_tokens=150)[0]["generated_text"]

        page_nums = {c["page_num"] for c in top_chunks}
        page_refs = ", ".join(f"Page {p}" for p in sorted(page_nums))

        st.markdown("### 🤖 Answer")
        st.write(f"{response.strip()}\n\n**Source(s):** {page_refs}")
