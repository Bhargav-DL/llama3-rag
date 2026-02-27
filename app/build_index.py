import fitz  
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

PDF_PATH = "data/tesla_manual.pdf"
INDEX_PATH = "vector_store/faiss.index"
CHUNKS_PATH = "vector_store/text_chunks.npy"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"

model = SentenceTransformer(EMBED_MODEL)

def load_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i+1, "text": text})
    return pages

def chunk_text(pages, chunk_size=500, overlap=100):
    chunks = []
    for page in pages:
        text = page["text"]
        page_num = page["page"]

        start = 0
        while start < len(text):
            chunk = text[start:start+chunk_size]
            chunks.append({"page": page_num, "text": chunk})
            start += chunk_size - overlap

    return chunks

print("Loading PDF...")
pages = load_pdf(PDF_PATH)

print("Chunking...")
chunks = chunk_text(pages)

texts = [c["text"] for c in chunks]

print("Embedding...")
embeddings = model.encode(texts, show_progress_bar=True)

print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

os.makedirs("vector_store", exist_ok=True)

faiss.write_index(index, INDEX_PATH)
np.save(CHUNKS_PATH, chunks)

print("Index created successfully!")