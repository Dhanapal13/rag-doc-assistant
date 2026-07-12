import asyncio
import json
from aiokafka import AIOKafkaConsumer
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "document-ingestion"

async def process_document(data: dict):
    try:
        logger.info(f"Processing job: {data['job_id']}")
        
        # Load document
        documents = SimpleDirectoryReader(input_files=[data["file_path"]]).load_data()
        
        # ChromaDB setup (same as your existing code)
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("rag_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create / update index
        index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
        
        logger.info(f"✅ Successfully indexed: {data['filename']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to process {data['job_id']}: {e}")

async def main():
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="rag-ingestion-group",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )
    await consumer.start()
    logger.info("✅ Ingestion Worker started")

    try:
        async for msg in consumer:
            await process_document(msg.value)
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(main())