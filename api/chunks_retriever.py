# find relevant chunks
from .pdf_ingestor import embedder, collections

def retrieve(query: str, n_results: int = 4) -> list[str]:
    qyr_embedding = embedder.encode([query]).tolist()
    result = collections.query(query_embeddings=qyr_embedding, n_results=n_results)
    return result["documents"][0]
