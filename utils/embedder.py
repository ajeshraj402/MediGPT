from sentence_transformers import SentenceTransformer
import torch

def chunk_text(pages, chunk_size=150):
    chunks = []
    for page in pages:
        words = page['text'].split()
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append({
                    'page_num': page['page_num'],
                    'chunk': chunk_text
                })
    return chunks

def get_embedder(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name, device="cpu")

def embed_chunks(chunks, model):
    texts = [chunk['chunk'] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings
