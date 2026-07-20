from typing import Any, Dict, List, Literal, Optional
import os
from dotenv import load_dotenv

from rag_evaluation import RagasEvaluator
from guardrails import InputGuardrail, OutputGuardrail
from rag_reranker import Reranker

load_dotenv()  # Load environment variables from .env file

# llama_index imports
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.sbert_rerank import SentenceTransformersRerank
from llama_index.llms.ollama import Ollama
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

class RAGService:
    def __init__(self):
        self.hf_index = None
        self.st_embedder = None
        self.st_client = None
        self.st_collection = None
        self.logger = structlog.get_logger()
        self._initialize_hf_llamaindex()
        self._initialize_sentence_transformer()
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

        index = self.get_hf_index()

        qa_template = PromptTemplate(
                    """Answer based ONLY on the context below. 
                        If you don't have enough information, say "I don't have enough information."

                        Context information is below:
                        ---------------------
                        {context_str}
                        ---------------------

                        Question: {query_str}
                        Answer: """
                                )
        
        rerank_postprocessor =  SentenceTransformersRerank(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=4)

        query_engine = index.as_query_engine(
            llm=Ollama(model= model == "llama" and LLAMA_MODEL or GEMMA_MODEL, 
                       text_qa_template=qa_template,
                       request_timeout=300, 
                       similarity_top_k=5),
                       nodes_postprocessors=[rerank_postprocessor]
        )
        with TOTAL_QUERY_LATENCY.labels(backend="llamaindex_hf", model="model").time():
            response = query_engine.query(query)

        answer_text = str(response)
        context_chunks = [str(node.get_content()) for node in response.source_nodes]

        output_result = self.output_guardrail.validate(answer_text, context_chunks)
        final_answer = output_result.redacted_text if output_result.redacted_text else answer_text

        return {
            "answer": final_answer,
            "context_chunks": context_chunks,
            "guardrail_flags": input_guardrail_result.flags if not input_guardrail_result.passed else None 
                                + output_result.flags if not output_result.passed else None,
            "blocked": not input_guardrail_result.passed or not output_result.passed,
        }        

    def _initialize_sentence_transformer(self):
        print("Initializing SentenceTransformer embedding model and ChromaDB client for RAGService")
        self.st_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.st_client = st_chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH","/app/chroma_db"))
        self.st_collection = self.st_client.get_or_create_collection("docs")

    def ingest_pdf_with_sentence_transformer(self, path: str, doc_id: str):
        print(f"Ingesting PDF with SentenceTransformer embedding: {path}")
        reader = SimpleDirectoryReader(input_dir=os.path.dirname(path))
        documents = reader.load_data()
        chunks = []
        for doc in documents:
            text = doc.get_text()
            for i in range(0, len(text), 400):
                chunks.append(text[i: i+400])
        
        embeddings = self.st_embedder.encode(chunks).tolist()
        self.st_collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        )
        print("PDF ingestion with SentenceTransformer embedding completed")

    def query_sentence_transformer(self, query: str, model: Literal["llama", "gemma"] = "llama",
                                    n_results: int = 4) -> List[str]:
        self.logger.info("rag_query_started", question_length=len(query), model=model)
        query_embedding = self.st_embedder.encode([query]).tolist()
        result = self.st_collection.query(query_embeddings=query_embedding, n_results=n_results)
        context_chunks = result["documents"][0] if result.get("documents") else []
        
        # rerank the context chunks using the Reranker class
        reranked_results = self.reranker.rerank(query, context_chunks, top_n=n_results)
        context_chunks_reranked = [chunk for chunk, score in reranked_results]
        context = "\n".join(context_chunks_reranked)

        ollama_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        prompt = f"""Answer based ONLY on the context below. If unsure, say no.
                    Context: {context}
                    Question: {query}"""
        
        with TOTAL_QUERY_LATENCY.labels(backend="sentence_transformer", model=model).time():
            response = ollama_client.chat(model= model == "llama" and LLAMA_MODEL or GEMMA_MODEL, 
                                          messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].split("\n")
    

    def evaluate_with_ragas(
        self,
        questions: List[str],
        model: Literal["llama", "gemma"] = "llama",
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the pipeline (retrieval + rerank + generation, guardrails included)
        over a batch of evaluation questions, then score the results with
        RAGAS. Use this offline (e.g. in CI or a nightly job) to track
        faithfulness / relevancy / precision / recall over time.
        """
        answers: List[str] = []
        contexts: List[List[str]] = []
 
        for question in questions:
            result = self.query_sentence_transformer(question, model=model)
            answers.append("\n".join(result["answer"]))
            contexts.append(result["context"])
 
        return self.ragas_evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
        )


rag_service = RAGService()

if __name__ == "__main__":    
    rag_service = RAGService()
    # rag_service.ingest_pdf_with_sentence_transformer("FG_DemandResponse.pdf", "ACER_Demand")
    rag_service.ingest_pdf_with_hf_llamaindex("FG_DemandResponse.pdf")

