from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import logging
from vector_db.qdrant import QdrantDatabaseConnector
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from settings import settings

logger = logging.getLogger(__name__)

class LongTermMemory:
    """
    Manage long-term memory for user context information over time.
    Use vector search to retrieve relevant information from conversation history.
    """
    
    def __init__(self, 
                 collection_name: str = "conversation_memory",
                 embedding_model: str = None,
                 max_context_length: int = 2048):
        """
        Args:
            collection_name: Name of the collection in Qdrant
            embedding_model: Model to create embeddings (default from settings)
            max_context_length: Maximum length of context when retrieving
        """
        self.collection_name = collection_name
        self.max_context_length = max_context_length
        
        # Initialize Qdrant connector
        self.qdrant = QdrantDatabaseConnector()
        
        # Initialize embedding model with safe loading
        embedding_model = embedding_model or settings.EMBEDDING_MODEL_ID
        try:
            from utils.model_utils import safe_load_sentence_transformer, get_safe_device
            
            # Get safe device
            safe_device = get_safe_device()
            logger.info(f"Using safe device: {safe_device}")
            
            # Load model safely
            self.embedder = safe_load_sentence_transformer(
                model_name=embedding_model,
                device=safe_device
            )
            
            if self.embedder is not None:
                logger.info(f"Initialized LongTermMemory with model: {embedding_model} on {safe_device}")
            else:
                raise RuntimeError("Failed to load embedding model with all strategies")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Set to None to disable long-term memory
            self.embedder = None
            logger.warning("LongTermMemory will operate in disabled mode (no vector search)")
        
        # Create collection if it doesn't exist (only if embedder is available)
        if self.embedder is not None:
            self._ensure_collection_exists()
        else:
            logger.warning("Skipping collection creation due to missing embedder")
    
    def _ensure_collection_exists(self) -> None:
        """Ensure collection exists in Qdrant."""
        try:
            self.qdrant.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except Exception:
            # Collection does not exist, create new one
            try:
                self.qdrant.create_vector_collection(self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to create collection {self.collection_name}: {e}")
                raise
    
    def store_message(self, 
                     session_id: str, 
                     message: str, 
                     role: str, 
                     metadata: Optional[Dict] = None,
                     user_id: Optional[str] = None) -> str:
        """
        Store message in long-term memory with vector embedding.
        
        Args:
            session_id: ID of the session
            message: Content of the message
            role: "user" or "assistant"
            metadata: Additional information
            user_id: ID of the user (to filter by user)
            
        Returns:
            point_id: ID of the point stored in Qdrant
        """
        # Check if embedder is available
        if self.embedder is None:
            logger.warning("Embedder not available, skipping message storage")
            return "disabled_mode"
            
        try:
            # Create embedding for the message
            vector = self.embedder.encode(message, show_progress_bar=False).tolist()
            
            # Prepare metadata
            timestamp = datetime.now()
            point_id = str(uuid.uuid4())
            
            payload = {
                "session_id": session_id,
                "user_id": user_id or "anonymous",
                "role": role,
                "message": message,
                "timestamp": timestamp.isoformat(),
                "message_length": len(message),
                **(metadata or {})
            }
            
            # Create point to store in Qdrant
            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            
            # Upsert into Qdrant
            self.qdrant._instance.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Stored message in long-term memory: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to store message in long-term memory: {e}")
            return "error"
    
    def retrieve_relevant_history(self, 
                                 query: str, 
                                 top_k: int = 5,
                                 user_id: Optional[str] = None,
                                 session_filter: Optional[List[str]] = None,
                                 time_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant history from long-term memory by semantic search.
        
        Args:
            query: Question/query to search for
            top_k: Number of results to return
            user_id: Filter by user_id (optional)
            session_filter: Filter by list of session_ids (optional)
            time_filter: Filter by time (optional)
            
        Returns:
            List of most relevant messages
        """
        # Check if embedder is available
        if self.embedder is None:
            logger.warning("Embedder not available, returning empty results")
            return []
            
        try:
            # Create embedding for the query
            query_vector = self.embedder.encode(query, show_progress_bar=False).tolist()
            
            # Create filter conditions
            filter_conditions = []
            
            if user_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                )
            
            if session_filter:
                filter_conditions.append(
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchAny(any=session_filter)
                    )
                )
            
            # Add time filter if present
            if time_filter:
                if "from_date" in time_filter:
                    filter_conditions.append(
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(gte=time_filter["from_date"])
                        )
                    )
            
            # Create filter object
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(must=filter_conditions)
            
            # Perform search
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
            
            # Extract and format results
            results = []
            for hit in search_results:
                result = {
                    "message": hit.payload["message"],
                    "role": hit.payload["role"],
                    "session_id": hit.payload["session_id"],
                    "timestamp": hit.payload["timestamp"],
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() 
                               if k not in ["message", "role", "session_id", "timestamp"]}
                }
                results.append(result)
            
            logger.debug(f"Retrieved {len(results)} relevant messages from long-term memory")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve from long-term memory: {e}")
            return []
    
    def get_user_context_summary(self, 
                                user_id: str, 
                                max_messages: int = 50) -> Dict[str, Any]:
        """
        Create a summary of the user's context from conversation history.
        
        Args:
            user_id: ID of the user
            max_messages: Maximum number of messages to analyze
            
        Returns:
            Dict containing summary information about the user
        """
        try:
            # Get all messages of the user (sorted by time)
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )
            
            # Scroll to get all messages of the user
            scroll_result = self.qdrant._instance.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=max_messages,
                with_payload=True
            )
            
            messages = [point.payload for point in scroll_result[0]]
            
            # Analyze and summarize
            total_messages = len(messages)
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            
            # Calculate statistics
            if messages:
                timestamps = [datetime.fromisoformat(msg["timestamp"]) for msg in messages]
                first_interaction = min(timestamps)
                last_interaction = max(timestamps)
                
                # Get unique session_ids
                unique_sessions = list(set(msg["session_id"] for msg in messages))
                
                summary = {
                    "user_id": user_id,
                    "total_messages": total_messages,
                    "user_messages": len(user_messages),
                    "assistant_messages": len(assistant_messages),
                    "unique_sessions": len(unique_sessions),
                    "first_interaction": first_interaction.isoformat(),
                    "last_interaction": last_interaction.isoformat(),
                    "session_ids": unique_sessions,
                    "recent_topics": self._extract_recent_topics(user_messages[-10:])  # 10 most recent messages
                }
                
                return summary
            else:
                return {"user_id": user_id, "total_messages": 0}
                
        except Exception as e:
            logger.error(f"Failed to get user context summary: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    def _extract_recent_topics(self, recent_messages: List[Dict]) -> List[str]:
        """
        Extract recent topics from the user's messages.
        Can be extended to use NLP techniques like keyword extraction.
        """
        # Simple: take the first 3-5 words of each message
        topics = []
        for msg in recent_messages:
            message_text = msg.get("message", "")
            words = message_text.split()[:5]  # Take first 5 words
            if words:
                topics.append(" ".join(words))
        
        return topics[:5]  # Return at most 5 topics
    
    def cleanup_old_data(self, days_threshold: int = 90) -> int:
        """
        Delete old data to save storage.
        
        Args:
            days_threshold: Number of days to determine old data
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_threshold)
            
            # Filter to find old data
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="timestamp",
                        range=models.Range(lt=cutoff_date.isoformat())
                    )
                ]
            )
            
            # Get list of point_ids to delete
            scroll_result = self.qdrant._instance.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1000,  # Delete in batches
                with_payload=False
            )
            
            point_ids = [point.id for point in scroll_result[0]]
            
            if point_ids:
                # Delete points
                self.qdrant._instance.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                
                logger.info(f"Cleaned up {len(point_ids)} old records from long-term memory")
                return len(point_ids)
            else:
                logger.info("No old records found to clean up")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about long-term memory."""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}