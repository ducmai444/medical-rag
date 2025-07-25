import concurrent.futures

import logger_utils
import utils
from vector_db.qdrant import QdrantDatabaseConnector
from qdrant_client import models
from rag.query_expansion import QueryExpansion
from rag.reranking import LLMReranker, HFCrossEncoderReranker
from rag.self_query import SelfQuery
from sentence_transformers.SentenceTransformer import SentenceTransformer
from settings import settings
import torch

logger = logger_utils.get_logger(__name__)


class VectorRetriever:
    def __init__(self, query: str) -> None:
        self._client = QdrantDatabaseConnector()
        self.query = query
        try:
            from utils.model_utils import safe_load_sentence_transformer, get_safe_device
            
            # Get safe device
            safe_device = get_safe_device()
            logger.info(f"Retriever: Using safe device: {safe_device}")
            
            # Load model safely
            self._embedder = safe_load_sentence_transformer(
                model_name=settings.EMBEDDING_MODEL_ID,
                device=safe_device
            )
            
            if self._embedder is not None:
                logger.info(f"Initialized VectorRetriever with model {settings.EMBEDDING_MODEL_ID} on {safe_device}")
            else:
                raise RuntimeError("Failed to load embedding model for retriever")
                
        except Exception as e:
            logger.exception(f"Failed to load SentenceTransformer: {str(e)}")
            raise
            
        self._query_expander = QueryExpansion()
        self._metadata_extractor = SelfQuery()
        self._reranker = LLMReranker()

    def _search_single_query(self, generated_query: str, metadata_filter_value: str, k: int):
        assert k >= 3, "k should be greater than 3"
        
        query_vector = self._embedder.encode(generated_query, show_progress_bar=False).tolist()
        if len(query_vector) != settings.EMBEDDING_SIZE:
            logger.error(f"Embedding size mismatch: expected {settings.EMBEDDING_SIZE}, got {len(query_vector)}")
            raise ValueError("Embedding size mismatch")

        collection_name = "vector_documents"
        try:
            search_result = self._client.search(
                collection_name=collection_name,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="author_id",
                            match=models.MatchValue(value=metadata_filter_value),
                        )
                    ]
                ),
                query_vector=query_vector,
                limit=k,
            )
            return search_result
        except Exception as e:
            logger.exception(f"Search failed for collection {collection_name}: {str(e)}")
            raise

    def retrieve_top_k(self, k: int, to_expand_to_n_queries: int) -> list:
        generated_queries = self._query_expander.generate_response(
            self.query, to_expand_to_n=to_expand_to_n_queries
        )
        logger.info("Successfully generated queries for search.", num_queries=len(generated_queries))

        author_id = self._metadata_extractor.generate_response(self.query)
        logger.info("Successfully extracted the author_id from the query.", author_id=author_id)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search_single_query, query, author_id, k)
                for query in generated_queries
            ]
            hits = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            hits = utils.flatten(hits)

        logger.info("All documents retrieved successfully.", num_documents=len(hits))
        return hits

    def rerank(self, hits: list, keep_top_k: int) -> list[str]:
        content_list = [hit.payload["content"] for hit in hits]
        rerank_hits = self._reranker.generate_response(
            query=self.query, passages=content_list, keep_top_k=keep_top_k
        )
        logger.info("Documents reranked successfully.", num_documents=len(rerank_hits))
        return rerank_hits

    def set_query(self, query: str):
        self.query = query