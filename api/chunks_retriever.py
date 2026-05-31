# find relevant chunks
from pdf_ingestor import embedder, collections

def retrieve(query: str, n_results: int = 4) -> list[str]:
    print(f"Retrieving chunks for query: {query}")
    query_embedding = embedder.encode([query]).tolist()
    if collections.count() == 0:
        print("No documents in collection. Returning empty list.")
    result = collections.query(query_embeddings=query_embedding, n_results=n_results)
    print(f"Retrieved chunks: {result['documents']}")
    return result["documents"][0]
