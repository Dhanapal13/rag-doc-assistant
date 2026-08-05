from typing import Any, Dict, List, Literal, Optional
import os
from dotenv import load_dotenv

from rag_evaluation import RagasEvaluator
from guardrails import InputGuardrail, OutputGuardrail
from rag_reranker import Reranker
from bm25_search import BM25Index, reciprocal_rank_fusion

load_dotenv()  # Load environment variables from .env file

# llama_index imports
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader, PromptTemplate
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
        self.hf_index = None
        self.st_embedder = None
        self.st_client = None
        self.st_collection = None
        self.bm25_index = BM25Index()
        self.logger = structlog.get_logger()
        self._initialize_hf_llamaindex()
        self.reranker = Reranker()
        self.input_guardrail = InputGuardrail(max_length=2000, min_length=3, blocklist_topics=settings.guardrail_blocklist_topics)
        self.output_guardrail = OutputGuardrail(max_length=2000, min_length=3, groundedness_threshold=settings.guardrail_groundedness_threshold)
        self.evaluator = RagasEvaluator(groundedness_threshold=settings.guardrail_groundedness_threshold)

    def _initialize_hf_llamaindex(self):
        """Force HuggingFace embedding model before any llama_index Settings access"""
        import llama_index.core
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        print("Forcing HuggingFace embedding model for llama_index")

        # This is the safest way - set it directly on the global Settings
        llama_index.core.Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model # "BAAI/bge-small-en-v1.5"
        )

    def ingest_pdf_with_hf_llamaindex(self, path: str):
        self._initialize_hf_llamaindex()
        print(f"Ingesting PDF with HuggingFace embedding for llama_index: {path}")
        chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", settings.chroma_db_path))
        chroma_collection = chroma_client.get_or_create_collection("hf_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        documents = SimpleDirectoryReader(input_files=[path]).load_data()
        VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, 
                                                embed_model=Settings.embed_model, show_progress=True)
        print("PDF ingestion with HuggingFace embedding for llama_index completed")

    def get_hf_index(self) -> VectorStoreIndex:
        self._initialize_hf_llamaindex()
        chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", settings.chroma_db_path))
        chroma_collection = chroma_client.get_or_create_collection("hf_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)
    
    def query_hf_index(self, query: str, model: Literal["llama", "gemma"] = "llama") -> List[str]:

        input_guardrail_result = self.input_guardrail.validate(query)
        if not input_guardrail_result.passed:
            self.logger.warning(
                "guardrail_blocked", layer="input", reason=input_guardrail_result.reason,
                query_preview=query[:80],
            )
            return [f"Input guardrail blocked the query: {input_guardrail_result.reason}"]

        # 1. Base retriever from your HF / vector store
        vectorstore = self.get_hf_index()          # must be a LangChain VectorStore
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 2. Reranker (same model you used)
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=4)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )

        # 3. LLM
        model_name = LLAMA_MODEL if model == "llama" else GEMMA_MODEL
        llm = ChatOllama(
            model=model_name,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            request_timeout=300,
        )

        # 4. Prompt (same content as your qa_template)
        prompt = ChatPromptTemplate.from_template(
            """Answer based ONLY on the context below. 
            If you don't have enough information, say "I don't have enough information."

            Context information is below:
            ---------------------
            {context}
            ---------------------

            Question: {input}
            Answer: """
                )

        # 5. Build the chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        with TOTAL_QUERY_LATENCY.labels(backend="llamaindex_hf", model=model).time():
            result = rag_chain.invoke({"input": query})

        answer_text = result["answer"]
        context_chunks = [doc.page_content for doc in result["context"]]

        output_result = self.output_guardrail.validate(answer_text, context_chunks)
        final_answer = output_result.redacted_text if output_result.redacted_text else answer_text

        combined_flags = list(input_guardrail_result.flags or []) + list(output_result.flags or [])

        return {
            "answer": final_answer,
            "context_chunks": context_chunks,
            "guardrail_flags": combined_flags or None,
            "blocked": not input_guardrail_result.passed or not output_result.passed,
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


rag_service_hf = RAGService()

if __name__ == "__main__":    
    rag_service_hf = RAGService()
    # rag_service.ingest_pdf_with_sentence_transformer("FG_DemandResponse.pdf", "ACER_Demand")
    rag_service_hf.ingest_pdf_with_hf_llamaindex("FG_DemandResponse.pdf")