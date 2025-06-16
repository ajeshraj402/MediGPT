import faiss
import json

def load_faiss_and_chunks(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve_top_k_chunks(question, model_embed, index, chunks, k=3):
    question_embedding = model_embed.encode([question], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, k)
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks