from typing import Any, Dict, List, Literal, Optional
import os
from dotenv import load_dotenv

from rag_evaluation import RagasEvaluator
from guardrails import InputGuardrail, OutputGuardrail
from rag_reranker import Reranker
from bm25_search import BM25Index, reciprocal_rank_fusion

load_dotenv()  # Load environment variables from .env file

# llama_index imports
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import chromadb

# Sentence transformer imports
from sentence_transformers import SentenceTransformer
import chromadb as st_chromadb
import ollama
from config import settings
from metrics import TOTAL_QUERY_LATENCY
import structlog

LLAMA_MODEL = settings.default_llm_model # "llama3.2:3b"
GEMMA_MODEL = settings.gemma_model

# Hybrid retrieval tuning. How many candidates each retriever pulls before
# fusion, and how those two ranked lists are weighted when merged with RRF.
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "15"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "15"))
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "1.0"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "1.0"))
FUSED_CANDIDATES_FOR_RERANK = int(os.getenv("FUSED_CANDIDATES_FOR_RERANK", "10"))


class RAGService:
    def __init__(self):
        self.st_embedder = None
        self.st_client = None
        self.st_collection = None
        self.bm25_index = BM25Index()
        self.logger = structlog.get_logger()
        self._initialize_sentence_transformer()
        self.reranker = Reranker()
        self.input_guardrail = InputGuardrail(max_length=2000, min_length=3, blocklist_topics=settings.guardrail_blocklist_topics)
        self.output_guardrail = OutputGuardrail(max_length=2000, min_length=3, groundedness_threshold=settings.guardrail_groundedness_threshold)
        self.evaluator = RagasEvaluator(groundedness_threshold=settings.guardrail_groundedness_threshold)
    
    def _initialize_sentence_transformer(self):
        print("Initializing SentenceTransformer embedding model, ChromaDB client, and BM25 index for RAGService")
        self.st_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        chroma_db_path = os.getenv("CHROMA_DB_PATH", "/app/chroma_db")
        self.st_client = st_chromadb.PersistentClient(path=chroma_db_path)
        self.st_collection = self.st_client.get_or_create_collection("docs")

        # Try to restore a previously persisted BM25 index so the lexical
        # side of hybrid search survives process restarts, same as Chroma.
        self.bm25_index_path = os.getenv(
            "BM25_INDEX_PATH", os.path.join(chroma_db_path, "bm25_index.pkl")
        )
        if not self.bm25_index.load(self.bm25_index_path):
            # No persisted index yet - rebuild it from whatever is already
            # in the Chroma "docs" collection, so BM25 and the vector store
            # stay in sync even if ingestion happened in an earlier run.
            existing = self.st_collection.get(include=["documents"])
            existing_ids = existing.get("ids") or []
            existing_docs = existing.get("documents") or []
            if existing_ids:
                self.bm25_index.build(existing_ids, existing_docs)
                self.bm25_index.save(self.bm25_index_path)

    def ingest_pdf_with_sentence_transformer(self, path: str, doc_id: str):
        print(f"Ingesting PDF with SentenceTransformer embedding: {path}")
        reader = SimpleDirectoryReader(input_dir=os.path.dirname(path))
        documents = reader.load_data()
        chunks = []
        for doc in documents:
            text = doc.get_text()
            for i in range(0, len(text), 400):
                chunks.append(text[i: i+400])

        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        # doc_id metadata is what lets us later find and delete every chunk
        # belonging to this document when it gets updated or removed.
        metadatas = [{"doc_id": doc_id} for _ in chunks]

        embeddings = self.st_embedder.encode(chunks).tolist()
        self.st_collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=chunk_ids,
            metadatas=metadatas,
        )

        # Keep the BM25 (lexical) index in lockstep with the vector store so
        # every chunk that's searchable by embedding similarity is also
        # searchable by exact/keyword match, and persist it to disk.
        self.bm25_index.add(chunk_ids, chunks)
        self.bm25_index.save(self.bm25_index_path)

        print("PDF ingestion with SentenceTransformer embedding (+ BM25 index update) completed")

    
    def _hybrid_retrieve(self, query: str, top_n: int) -> List[str]:
        """
        Hybrid retrieval: run dense vector search and BM25 lexical search in
        parallel, fuse the two ranked id lists with Reciprocal Rank Fusion,
        take the top fused candidates, then cross-encoder rerank down to
        top_n chunks of context text for the LLM.
        """
        # 1. Dense vector retrieval (bi-encoder, fast, semantic).
        query_embedding = self.st_embedder.encode([query]).tolist()
        vector_result = self.st_collection.query(
            query_embeddings=query_embedding, n_results=VECTOR_TOP_K
        )
        vector_ids = (vector_result.get("ids") or [[]])[0]
        vector_docs = (vector_result.get("documents") or [[]])[0]
        text_by_id: Dict[str, str] = dict(zip(vector_ids, vector_docs))

        # 2. Sparse lexical retrieval (BM25, catches exact keyword matches).
        bm25_hits = self.bm25_index.search(query, top_k=BM25_TOP_K)
        bm25_ids = [chunk_id for chunk_id, _text, _score in bm25_hits]
        for chunk_id, text, _score in bm25_hits:
            text_by_id.setdefault(chunk_id, text)

        # 3. Fuse both ranked lists by rank (not raw score) via RRF.
        fused = reciprocal_rank_fusion(
            [vector_ids, bm25_ids],
            k=RRF_K,
            weights=[VECTOR_WEIGHT, BM25_WEIGHT],
        )
        fused_candidate_ids = [chunk_id for chunk_id, _score in fused[:FUSED_CANDIDATES_FOR_RERANK]]
        fused_candidate_texts = [text_by_id[cid] for cid in fused_candidate_ids if cid in text_by_id]

        if not fused_candidate_texts:
            return []

        # 4. Cross-encoder rerank the fused shortlist down to top_n.
        return self.reranker.rerank_texts_only(query, fused_candidate_texts, top_n=top_n)

    def query_sentence_transformer(self, query: str, model: Literal["llama", "gemma"] = "llama",
                                    n_results: int = 4) -> Dict[str, Any]:
        self.logger.info("rag_query_started", question_length=len(query), model=model)

        input_guardrail_result = self.input_guardrail.validate(query)
        if not input_guardrail_result.passed:
            self.logger.warning(
                "guardrail_blocked", layer="input", reason=input_guardrail_result.reason,
                query_preview=query[:80],
            )
            return {
                "answer": f"Input guardrail blocked the query: {input_guardrail_result.reason}",
                "context_chunks": [],
                "guardrail_flags": input_guardrail_result.flags,
                "blocked": True,
            }

        context_chunks = self._hybrid_retrieve(query, top_n=n_results)
        context = "\n".join(context_chunks)

        model_name = model == "llama" and LLAMA_MODEL or GEMMA_MODEL
        llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        prompt = f"""Answer based ONLY on the context below. If unsure, say no.
                    Context: {context}
                    Question: {query}"""

        with TOTAL_QUERY_LATENCY.labels(backend="sentence_transformer_hybrid", model=model).time():
            response = llm.invoke([HumanMessage(content=prompt)])

        answer_text = response["message"]["content"]

        output_result = self.output_guardrail.validate(answer_text, context_chunks)
        final_answer = output_result.redacted_text if output_result.redacted_text else answer_text

        combined_flags = list(input_guardrail_result.flags or []) + list(output_result.flags or [])

        return {
            "answer": final_answer,
            "context_chunks": context_chunks,
            "guardrail_flags": combined_flags or None,
            "blocked": not output_result.passed,
        }
    

    def evaluate_with_ragas(
        self,
        questions: List[str],
        model: Literal["llama", "gemma"] = "llama",
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the pipeline (hybrid retrieval + rerank + generation, guardrails
        included) over a batch of evaluation questions, then score the
        results with RAGAS. Use this offline (e.g. in CI or a nightly job)
        to track faithfulness / relevancy / precision / recall over time.
        """
        answers: List[str] = []
        contexts: List[List[str]] = []

        for question in questions:
            result = self.query_sentence_transformer(question, model=model)
            answers.append(result["answer"])
            contexts.append(result["context_chunks"])

        return self.evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
        )

    def delete_document(self, doc_id: str) -> int:
            """
            Remove every chunk belonging to doc_id from BOTH the vector store
            and the BM25 index. Call this before re-ingesting an updated
            version of a document, or to retire a document entirely.
            Returns the number of chunks removed.
            """
            existing = self.st_collection.get(where={"doc_id": doc_id}, include=[])
            chunk_ids = existing.get("ids") or []
            if not chunk_ids:
                self.logger.info("delete_document_noop", doc_id=doc_id)
                return 0
    
            self.st_collection.delete(ids=chunk_ids)
            self.bm25_index.remove(chunk_ids)
            self.bm25_index.save(self.bm25_index_path)
    
            self.logger.info("delete_document_completed", doc_id=doc_id, chunks_removed=len(chunk_ids))
            return len(chunk_ids)
    
    def update_document(self, path: str, doc_id: str) -> None:
        """
        Re-ingest a document that has changed on disk: delete its old
        chunks from both indexes, then chunk/embed/index the new version
        under the same doc_id. Simplest correct way to handle frequently
        changing source documents - see the docstring note below for a
        cheaper incremental alternative when only part of a document
        changes and re-embedding the whole thing is too expensive.
        """
        removed = self.delete_document(doc_id)
        self.logger.info("update_document_reingesting", doc_id=doc_id, old_chunks_removed=removed)
        self.ingest_pdf_with_sentence_transformer(path, doc_id)
    

rag_service_sf = RAGService()

if __name__ == "__main__":    
    rag_service_sf = RAGService()
    # rag_service_sf.ingest_pdf_with_sentence_transformer("FG_DemandResponse.pdf", "ACER_Demand")
    rag_service_sf.ingest_pdf_with_hf_llamaindex("FG_DemandResponse.pdf")