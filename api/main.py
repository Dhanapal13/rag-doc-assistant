# Fast API Services
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
def ask(requet: QuestionRequest):    
    return {
        "answer": f"you asked {requet.question}"
    }

