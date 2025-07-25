import logger_utils
from typing import Optional, List
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Batch, Distance, VectorParams
from sentence_transformers import SentenceTransformer
from utils.chunking import chunk_text
import torch
import os
import uuid

from settings import settings

logger = logger_utils.get_logger(__name__)


class QdrantDatabaseConnector:
    _instance: Optional[QdrantClient] = None

    def __init__(self) -> None:
        if self._instance is None:
            try:
                if settings.USE_QDRANT_CLOUD:
                    self._instance = QdrantClient(
                        url=settings.QDRANT_CLOUD_URL,
                        api_key=settings.QDRANT_APIKEY,
                    )
                else:
                    self._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT,
                    )
            except UnexpectedResponse:
                logger.exception(
                    "Couldn't connect to Qdrant.",
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                    url=settings.QDRANT_CLOUD_URL,
                )

                raise

    def get_collection(self, collection_name: str):
        return self._instance.get_collection(collection_name=collection_name)

    def create_non_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name, vectors_config={}
        )

    def create_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_SIZE,
                distance=Distance.COSINE
            ),
        )
        
        try:
            self._instance.create_payload_index(
                collection_name=collection_name,
                field_name="author_id",
                field_type=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created index for 'author_id' field in collection {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to create index for 'author_id': {str(e)}")

    def write_data(self, collection_name: str, points: Batch):
        try:
            self._instance.upsert(collection_name=collection_name, points=points)
        except Exception:
            logger.exception("An error occurred while inserting data.")

            raise
        
    def search(self, collection_name: str, query_vector: list, query_filter: models.Filter, limit: int) -> list:
        return self._instance.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def scroll(self, collection_name: str, limit: int):
        return self._instance.scroll(collection_name=collection_name, limit=limit)

    def close(self):
        if self._instance:
            self._instance.close()

            logger.info("Connected to database has been closed.")

    def store_text_with_chunking(
        self,
        collection_name: str,
        text: str,
        metadata: dict,
        batch_size: int = 100
    ) -> None:
        if not text or not isinstance(text, str):
            logger.error("Invalid text input", collection_name=collection_name)
            raise ValueError("Text input must be a non-empty string")
        
        if not metadata or not isinstance(metadata, dict):
            logger.error("Invalid metadata input", collection_name=collection_name)
            raise ValueError("Metadata must be a non-empty dictionary")

        # Set environment variable to avoid memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Chunk the text
        chunks = chunk_text(text)
        if not chunks:
            logger.warning("No chunks generated", collection_name=collection_name)
            return

        # Initialize SentenceTransformer với safe loading
        try:
            from utils.model_utils import safe_load_sentence_transformer, get_safe_device
            
            # Get safe device
            safe_device = get_safe_device()
            logger.info(f"Qdrant: Using safe device: {safe_device}")
            
            # Load model safely
            embedder = safe_load_sentence_transformer(
                model_name=settings.EMBEDDING_MODEL_ID,
                device=safe_device
            )
            
            if embedder is None:
                raise RuntimeError("Failed to load embedding model for Qdrant operations")
                
            logger.info(f"Using {safe_device} for embedding generation")
            device = safe_device
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise RuntimeError(f"Cannot initialize embedding model: {e}")

        # Generate embeddings for chunks
        try:
            embeddings = embedder.encode(
                chunks,
                batch_size=16,
                show_progress_bar=False,
                device=device
            ).tolist()
        except Exception as e:
            logger.exception(f"Failed to generate embeddings: {str(e)}")
            if "CUDA out of memory" in str(e) and device == "cuda":
                logger.info("Falling back to CPU for embedding generation")
                embedder.to("cpu")
                embeddings = embedder.encode(
                    chunks,
                    batch_size=16,
                    show_progress_bar=False,
                    device="cuda"
                ).tolist()
            else:
                raise

        # Prepare points for batch insertion with UUID
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())  # Tạo UUID cho point ID
            point = models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": chunk,
                    "chunk_index": i,
                    "original_id": metadata.get('id', 'doc'),  # Lưu ID gốc trong payload
                    **metadata
                }
            )
            points.append(point)

        # Batch upsert points
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            try:
                self._instance.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
                logger.info(
                    "Stored batch of chunks",
                    batch_size=len(batch_points),
                    collection_name=collection_name
                )
            except UnexpectedResponse as e:
                logger.exception(
                    "Failed to store batch of chunks",
                    batch_index=i // batch_size,
                    error=str(e)
                )
                raise

        logger.info(
            "Successfully stored text chunks",
            num_chunks=len(chunks),
            collection_name=collection_name
        )