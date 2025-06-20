import os
import json
import faiss
import streamlit as st
from tempfile import NamedTemporaryFile

from utils.pdf_parser import extract_text_by_page
from utils.embedder import chunk_text, get_embedder, embed_chunks
from utils.retriever import load_faiss_and_chunks, retrieve_top_k_chunks
from utils.rag_engine import get_flan_pipeline, build_prompt

qa_pipeline = get_flan_pipeline()
model_embed = get_embedder()

st.set_page_config(page_title="MediPDF Chatbot", page_icon="🩺")
st.title("Medical PDF Chatbot with Page Sources")
st.markdown("Upload a medical related PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        pdf_path = temp_pdf.name

    st.success("PDF uploaded successfully!")

    # Extract and process
    pages = extract_text_by_page(pdf_path)
    chunks = chunk_text(pages, chunk_size=150)
    embeddings = embed_chunks(chunks, model_embed)

    # Save FAISS index to memory
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save chunks in-memory
    st.session_state.chunks = chunks
    st.session_state.index = index

if "chunks" in st.session_state and "index" in st.session_state:
    question = st.text_input("Ask a question about the PDF:")

    if question:
        top_chunks = retrieve_top_k_chunks(
            question, model_embed, st.session_state.index, st.session_state.chunks, k=3
        )

        prompt, _ = build_prompt(question, top_chunks)
        response = qa_pipeline(prompt, max_new_tokens=150)[0]['generated_text']

        page_nums = set(c['page_num'] for c in top_chunks)
        page_refs = ", ".join(f"Page {p}" for p in sorted(page_nums))

        final_answer = f"{response.strip()}\n **Source(s):** {page_refs}"
        st.markdown("### Answer")
        st.write(final_answer)
