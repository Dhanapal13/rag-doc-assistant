# Fast API Services 
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.rag_service import rag_service
# Add App object
app = FastAPI(title="RAG Document Assistant")

# enable CORS
app.add_middleware(CORSMiddleware, 
                   allow_origins=["*"], 
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )

class QuestionRequest(BaseModel):
    question: str
    backend: Literal["st", "hf"] = "st"  # Default to HuggingFace if not specified
    model: Literal["llama", "gemma"] = "llama"  # Default LLM model to llama

@app.get("/")
def root():
    return "Welcome to RAG DOC System!!!"

@app.post("/ask")
def ask(request: QuestionRequest):
    if request.backend == "st":
        result = rag_service.query_sentence_transformer(request.question, request.model)
    else:
        result = rag_service.query_hf_index(request.question, request.model)

    return {
        "answer": result, "backend": request.backend, "model": request.model
    }

