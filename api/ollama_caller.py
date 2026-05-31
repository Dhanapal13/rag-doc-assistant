import os
import ollama
from chunks_retriever import retrieve

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_client = ollama.Client(host=OLLAMA_BASE_URL)

def rag_answer(query: str) -> str:
    context_chunks = retrieve(query)
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer based ONLY on the context below. If unsure, say no.
                Context: {context}
                Question: {query}"""
    print("Prompt to LLM:", prompt)
    response = _client.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
    print("Response from LLM:", response["message"]["content"])
    return response["message"]["content"]