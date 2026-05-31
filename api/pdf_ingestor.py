# PDF chunking and embeddings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer   
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH","/app/chroma_db"))
collections = client.get_or_create_collection("docs")

def ingest_pdf(path:str, doc_id: str):
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        # simple 500 char overlapping text
        for i in range(0, len(text), 400):
            chunks.append(text[i: i+400])
    
    embeddings = embedder.encode(chunks).tolist()
    collections.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    )

if __name__ == "__main__":
    path = "FG_DemandResponse.pdf"
    print("Ingest ACER PDF")
    ingest_pdf(path=path, doc_id="ACER_Demand_")