# \# Tesla Manual RAG Assistant (Llama-3 + FAISS + FastAPI)

# 

# \## Overview

# 

# This project implements a \*\*Retrieval-Augmented Generation (RAG)\*\* system that allows users to ask technical questions about a Tesla Owner’s Manual and receive accurate answers grounded in the document.

# The system uses:

# \* \*\*Llama-3 (8B Instruct)\*\* for response generation

# \* \*\*FAISS\*\* for fast vector similarity search

# \* \*\*BGE embeddings\*\* for high-quality semantic retrieval

# \* \*\*FastAPI\*\* for production-ready API deployment

# The assistant answers questions using only the manual content and returns \*\*source page references\*\*.

# \## Features

# \* Ask technical questions about Tesla features and safety

# \* Context-aware answers from a 300+ page manual

# \* Source page citations

# \* Fast semantic search using FAISS

# \* Production-ready FastAPI backend

# \* Config-driven architecture

# \* GPU-ready (cloud deployment supported)

# \## Project Structure

# tesla-rag/

# │

# ├── app/

# │   ├── main.py            # FastAPI application

# │   ├── config.py          # Model and path configuration

# │   ├── rag\_pipeline.py    # Retrieval + generation logic

# │   └── build\_index.py     # Script to create FAISS index

# │

# ├── data/

# │   └── tesla\_manual.pdf

# │

# ├── vector\_store/

# │   └── .gitkeep           # FAISS files generated locally

# │

# ├── requirements.txt

# ├── README.md

# └── .gitignore

# \## How It Works

# 1\. PDF is split into text chunks

# 2\. Chunks are embedded using \*\*BGE-large\*\*

# 3\. Embeddings stored in \*\*FAISS\*\*

# 4\. User query → embedded → top relevant chunks retrieved

# 5\. Retrieved context → sent to \*\*Llama-3\*\*

# 6\. Model generates grounded answer with page references

# 

# Architecture:

# User → FastAPI → FAISS → Context → Llama-3 → Response

# ```

# \## Setup Instructions

# 

# \### 1. Install dependencies

# pip install -r requirements.txt

# \### 2. Build the Vector Index (Required)

# Before running the API:

# python app/build\_index.py

# This will generate:

# vector\_store/faiss.index

# vector\_store/text\_chunks.npy

# Note: These files are not included in the repository.

# \### 3. Run FastAPI

# uvicorn app.main:app --host 0.0.0.0 --port 8000

# Open Swagger UI:

# http://localhost:8000/docs

# \## GPU Requirement
# This project uses:

# \*\*meta-llama/Meta-Llama-3-8B-Instruct\*\*
# Recommended hardware:
# \* GPU with \*\*16GB+ VRAM\*\*

# \* Or deploy on:

# 

# &nbsp; \* RunPod

# &nbsp; \* Vast.ai

# &nbsp; \* AWS GPU instances

# For development/testing, cloud GPU is recommended.

# \## API Example

# \*\*POST /ask\*\*

# Request:

# 

# ```json

# {

# &nbsp; "question": "How do I use Autopark?"

# }

# ```

# 

# Response:

# 

# ```json

# {

# &nbsp; "question": "...",

# &nbsp; "answer": "...",

# &nbsp; "sources": \[138, 139]

# }

# ```
# This system is designed for production deployment using:
# \* FastAPI + Uvicorn

# \* Docker (optional)

# \* Cloud GPU platforms (RunPod / AWS / GCP)

# \## Notes

# \* FAISS index must be generated locally using `build\_index.py`

# \* Llama-3 requires Hugging Face access approval

# \* Generated files in `vector\_store/` are ignored by Git

