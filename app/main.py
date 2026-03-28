from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache

from fastapi.middleware.cors import CORSMiddleware

from app.ingest import load_pdfs, chunk_docs
from app.retriever import HybridRetriever
from app.rag_pipeline import generate_answer
from app.verifier import verify_answer
from app.memory import Memory

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str


@lru_cache()
def init_system():
    pdfs = [
        "data/pdfs/doc1.pdf",
        "data/pdfs/doc2.pdf",
        "data/pdfs/doc3.pdf"
    ]

    docs = load_pdfs(pdfs)
    chunks = chunk_docs(docs)

    retriever = HybridRetriever(chunks)
    memory = Memory()

    return retriever, memory


@app.get("/")
def home():
    return {"status": "RAG running"}


@app.post("/query")
def query(data: Query):
    retriever, memory = init_system()

    docs = retriever.retrieve(data.query)

    docs = docs[:3]

    context = "\n\n".join([d.page_content for d in docs])

    full_context = memory.get() + "\n\n" + context

    answer = generate_answer(data.query, full_context)
    verified = verify_answer(answer, context)

    memory.add(data.query, answer)

    return {
        "answer": answer,
        "verified": verified
    }