import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import (
    LLM_MODEL_NAME,
    EMBED_MODEL_NAME,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    MAX_NEW_TOKENS,
    TOP_K_RETRIEVAL,
    SYSTEM_PROMPT
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

def load_vector_store():
    if not os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index not found")
        return None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    chunks = np.load(CHUNKS_PATH, allow_pickle=True)
    return index, chunks

index, chunks = load_vector_store()

tokenizer = None
model = None

def load_llm():
    print("Loading Llama-3...")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        use_auth_token=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model

def retrieve(query):
    if index is None:
        return []

    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb), TOP_K_RETRIEVAL)

    return [chunks[i] for i in I[0]]

def generate_answer(query: str):
    global tokenizer, model

    if tokenizer is None or model is None:
        tokenizer, model = load_llm()

    docs = retrieve(query)

    if not docs:
        return "Vector store not found.", []

    context = "\n\n".join(
        [f"(Page {d['page']}) {d['text']}" for d in docs]
    )

    user_message = f"""
Context:
{context}

Question: {query}
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in decoded:
        decoded = decoded.split("assistant")[-1].strip()

    pages = [{"page": d["page"]} for d in docs]

    return decoded, pages