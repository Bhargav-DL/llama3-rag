from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import generate_answer

app = FastAPI(title="Tesla RAG API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "Tesla RAG API running"}

@app.post("/ask")
def ask(request: QueryRequest):
    answer, sources = generate_answer(request.question)
    
    pages = sorted(list(set([s["page"] for s in sources])))

    return {
        "question": request.question,
        "answer": answer,
        "sources": pages
    }