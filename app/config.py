
# Model Configuration

LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"

FAISS_INDEX_PATH = "vector_store/faiss.index"
CHUNKS_PATH = "vector_store/text_chunks.npy"

PDF_PATH = "data/tesla_manual.pdf"

# Generation Settings

MAX_NEW_TOKENS = 250
TOP_K_RETRIEVAL = 3

# System Prompt

SYSTEM_PROMPT = """
You are a Tesla Technical Expert.

Instructions:
- Answer using ONLY the provided context
- Be concise and professional
- Do not repeat information
- If the answer is not available, say:
"The information is not available in the manual."
"""