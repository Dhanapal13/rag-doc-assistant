# Fast API Services 
from typing import Literal
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.rag_service_sf import rag_service_sf
from api.rag_service_hf import rag_service_hf

from config import settings
from middleware import RequestIDMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from metrics import RAG_QUERIES_TOTAL
import structlog
from contextlib import asynccontextmanager
from aiokafka import AIOKafkaProducer


KAFKA_BOOTSTRAP = "localhost:9092"   # or broker:29092 inside Docker
INGESTION_TOPIC = "document-ingestion"

producer: AIOKafkaProducer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global producer
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP, value_serializer=lambda v: v.encode('utf-8'))
    await producer.start()
    print("Kafka producer started")
    try:
        yield
    finally:
        await producer.stop()
        print("Kafka producer stopped")


# Add App object
app = FastAPI(title= settings.app_name)
logger = structlog.get_logger()
app.add_middleware(RequestIDMiddleware)
# enable CORS
app.add_middleware(CORSMiddleware, 
                   allow_origins=["*"], 
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )


class QuestionRequest(BaseModel):
    question: str
    backend: Literal["st", "hf"] = "st"  # Default to HuggingFace if not specified
    model: Literal["llama", "gemma"] = "llama"  # Default LLM model to llama

class IngestionResponse(BaseModel):
    message: str
    status: str
    job_id: str = None  # Optional field for ingestion ID


@app.get("/")
def root():
    return "Welcome to RAG DOC System!!!"


@app.post("/ingest", response_model=IngestionResponse)
async def ingest(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    file_path = f"data/{job_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    event = {
        "job_id": job_id,
        "file_path": file_path,
        "filename": file.filename,
        "metadata": {
            "source": "upload",
        }
    }

    await producer.send_and_wait(INGESTION_TOPIC, str(event))

    return IngestionResponse(message="File uploaded successfully", status="queued", job_id=job_id)


@app.post("/ask")
def ask(request: QuestionRequest):

    if request.backend == "st":
        result = rag_service_sf.query_sentence_transformer(request.question, request.model)
    else:
        result = rag_service_hf.query_hf_index(request.question, request.model)

    return {
        "answer": result, "backend": request.backend, "model": request.model
    }

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


