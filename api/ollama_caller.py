# Prompt assembly and Ollama calls
import ollama
from .chunks_retriever import retrieve

def rag_answer(query: str) -> str:
    context_chunks = retrieve(query)
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer based ONLY on the context below. If unsure, say no.
        Context: {context} 
        Question: {query}"""
    response = ollama.chat(model="llama3.2:3b", messages=[{"role":"user", "content": prompt}])
    return response["message"]["content"]

