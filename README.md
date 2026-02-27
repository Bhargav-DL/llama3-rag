# Tesla Manual RAG Assistant (Llama-3 + FAISS + FastAPI)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask technical questions about a Tesla Owner’s Manual and receive accurate, context-grounded answers.

The system retrieves relevant sections from the document using semantic search and generates responses using Llama-3. Each answer includes source page references for verification.

---

## Key Features

* Context-aware answers grounded in the manual
* Source page citations
* Fast semantic retrieval using FAISS
* Config-based architecture
* FastAPI backend for production deployment
* Designed for cloud GPU environments

---

## System Architecture

```
User Query
   ↓
FastAPI API
   ↓
Embedding (BGE)
   ↓
FAISS Vector Search
   ↓
Relevant Context
   ↓
Llama-3 Generation
   ↓
Final Answer + Sources
```

---

## Project Structure

```
tesla-rag/
│
├── app/
│   ├── main.py            # FastAPI application
│   ├── config.py          # Model and path configuration
│   ├── rag_pipeline.py    # Retrieval and generation logic
│   └── build_index.py     # FAISS index creation script
│
├── data/
│   └── tesla_manual.pdf
│
├── vector_store/
│   └── .gitkeep           # Index files generated locally
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Build the vector index

Run this once to generate embeddings and FAISS index:

```
python app/build_index.py
```

This will create:

* `vector_store/faiss.index`
* `vector_store/text_chunks.npy`

---

### 3. Run the API

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```
http://localhost:8000/docs
```

---

## API Example

**POST /ask**

Request:

```json
{
  "question": "How do I use Autopark?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [138, 139]
}
```

---

## Model Requirements

This project uses:

* meta-llama/Meta-Llama-3-8B-Instruct
* BAAI/bge-large-en-v1.5

Recommended environment:

* GPU with 16GB+ VRAM
* Or cloud deployment (RunPod, AWS, Vast.ai)

---

## Use Cases

* Technical documentation assistants
* Customer support automation
* Enterprise knowledge base search
* Internal document QA systems

# \* Llama-3 requires Hugging Face access approval

# \* Generated files in `vector\_store/` are ignored by Git

