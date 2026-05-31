# Fast API Services
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama_caller import rag_answer

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

@app.get("/")
def root():
    return "Welcome to RAG DOC System!!!"

@app.post("/ask")
def ask(request: QuestionRequest):    
    result = rag_answer(request.question)
    return {
        "answer": {result}
    }

