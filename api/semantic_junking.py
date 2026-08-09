"""
RAG with sentence-transformers
- Semantic (context-aware) chunking
- High-quality local embeddings
- ChromaDB vector store
- Optional: Ollama for generation
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.config import Settings
import re
from typing import List

# ============================================================
# 1. Load Embedding Model
# ============================================================
# Good options:
#   "all-MiniLM-L6-v2"          → very fast, 384 dims (good baseline)
#   "nomic-ai/nomic-embed-text-v1.5"  → strong general model (768 dims)
#   "BAAI/bge-m3"               → excellent multilingual + long context
#   "BAAI/bge-large-en-v1.5"    → high English quality

print("Loading embedding model...")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")   # or "all-MiniLM-L6-v2"

def embed(texts: List[str] | str) -> np.ndarray:
    """Embed one or many texts. Returns normalized vectors."""
    return model.encode(
        texts,
        normalize_embeddings=True,   # enables fast cosine via dot product
        show_progress_bar=False
    )


# ============================================================
# 2. Semantic / Context-aware Chunking
# ============================================================
def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.72,
    max_chunk_chars: int = 900,
    min_chunk_chars: int = 120
) -> List[str]:
    """
    Split text into coherent chunks based on semantic similarity
    between consecutive sentences.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return []

    # Embed all sentences at once (very fast)
    embeddings = embed(sentences)

    chunks = []
    current = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        # Cosine similarity (since vectors are normalized → just dot product)
        sim = float(np.dot(embeddings[i-1], embeddings[i]))

        sent = sentences[i]
        sent_len = len(sent)

        # Decide whether to start a new chunk
        too_long = current_len + sent_len > max_chunk_chars
        topic_changed = sim < similarity_threshold

        if (topic_changed or too_long) and current_len >= min_chunk_chars:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


# ============================================================
# 3. Build Vector Store
# ============================================================
def create_vector_store(documents: List[str]):
    client = chromadb.PersistentClient(path="./st_rag_db")
    collection = client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    all_ids = []
    all_embeddings = []

    for doc_idx, doc in enumerate(documents):
        chunks = semantic_chunk(doc)
        print(f"Document {doc_idx+1} → {len(chunks)} semantic chunks")

        # Batch embed all chunks of this document
        if chunks:
            embs = embed(chunks)
            for i, (chunk, emb) in enumerate(zip(chunks, embs)):
                all_chunks.append(chunk)
                all_ids.append(f"doc{doc_idx}_chunk{i}")
                all_embeddings.append(emb.tolist())

    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_chunks
    )
    print(f"Total chunks stored: {len(all_chunks)}")
    return collection


# ============================================================
# 4. Query + Optional Generation with Ollama
# ============================================================
def rag_query(collection, question: str, top_k: int = 3, use_ollama: bool = True):
    # Embed the question with the same model
    q_emb = embed(question).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    retrieved = results["documents"][0]
    context = "\n\n".join(retrieved)

    print("\n--- Retrieved Chunks ---")
    for i, c in enumerate(retrieved, 1):
        print(f"[{i}] {c[:150]}...")
    print("------------------------\n")

    if not use_ollama:
        return context   # just return the context

    # Generate answer with local Ollama LLM
    import ollama
    prompt = f"""Answer the question using only the context below.
If the answer is not present, say "I don't know based on the provided information."

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2",          # change to your preferred model
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    docs = [
        """
        Ollama is a tool that makes it easy to run large language models locally.
        You can download models with a single command like ollama pull llama3.2.
        It supports both chat models and dedicated embedding models.
        The library exposes a clean Python API and a REST endpoint on port 11434.
        """,
        """
        Retrieval-Augmented Generation (RAG) combines a retriever with a generator.
        The quality of the chunks is often more important than the embedding model itself.
        Semantic chunking detects topic boundaries by measuring similarity between sentences.
        This produces more coherent pieces of text for the language model to use.
        sentence-transformers is one of the most popular libraries for local embeddings.
        """
    ]

    print("Building vector store with semantic chunking...")
    collection = create_vector_store(docs)

    questions = [
        "How do I download a model in Ollama?",
        "Why is semantic chunking useful in RAG?",
        "What library is commonly used for local embeddings?"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        answer = rag_query(collection, q, use_ollama=True)
        print(f"\nAnswer:\n{answer}")