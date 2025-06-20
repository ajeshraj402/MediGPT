# Medical PDF Chatbot with Page Source Tracking

![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)
![Model](https://img.shields.io/badge/Model-Flan--T5--Base-blue)
![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM-lightgrey)
![Vector Search](https://img.shields.io/badge/FAISS-Semantic%20Search-yellow)

This project is a **RAG (Retrieval-Augmented Generation)** chatbot that allows users to:
- Upload **medical-related PDF documents** (like Medicare guides, FDA sheets, insurance docs)
- Ask **natural language questions**
- Get answers **based only on the PDF content**
- See **page numbers** where the answer was found
- All hosted in a **clean Streamlit app**

---

## Use Cases
- Understanding health insurance coverage
- Reading through lengthy government medical booklets
- Finding drug instructions or treatment guidelines in large documents

---

## Demo
Try it live (free):  https://medigpt.streamlit.app/


---

## How It Works

- **Text Extraction**: Extracts text page-by-page using PyMuPDF
- **Chunking & Embedding**: Uses `all-MiniLM-L6-v2` to embed ~150-word chunks
- **Semantic Search**: Finds similar chunks to your question using FAISS
- **Answer Generation**: Uses `flan-t5-base` to answer based on retrieved context
- **Page References**: Automatically includes source pages for transparency
- **Medical Disclaimer**: Reminds users it's not medical advice

---

## Tech Stack

| Layer              | Tool                                |
|--------------------|-------------------------------------|
| Embeddings         | `sentence-transformers` (MiniLM)    |
| Vector Search      | `faiss-cpu`                         |
| Language Model     | `google/flan-t5-base` via `transformers` |
| UI                 | `Streamlit`                         |
| PDF Parser         | `PyMuPDF`                           |

---

## Installation (Run Locally)

```bash
git clone https://github.com/ajeshraj402/MediGPT
pip install -r requirements.txt
streamlit run app.py
