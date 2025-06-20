import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


HF_TOKEN = (
    st.secrets["huggingface"]["token"]
    if "huggingface" in st.secrets
    else os.getenv("HF_TOKEN")
)


def get_flan_pipeline():
    model_id = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        token=HF_TOKEN,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        token=HF_TOKEN
    )

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
    )


def build_prompt(question, top_chunks):
    context = "\n\n".join(
        f"(Page {chunk['page_num']}): {chunk['chunk']}" for chunk in top_chunks
    )

    prompt = f"""You are a medical document assistant.

Use the context below to answer the user's question. You must:
- Answer ONLY using the context.
- Clearly mention the relevant page numbers (e.g., "As stated on Page 7…").
- Do NOT make up or guess anything.
- Add a medical disclaimer: "This is not medical advice."

Context:
{context}

Question: {question}

Answer:"""

    return prompt, context
