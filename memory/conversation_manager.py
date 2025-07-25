# memory/conversation_manager.py

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages merging context from short-term and long-term memory,
    creates input context for the inference pipeline.
    """
    
    def __init__(self, 
                 max_context_tokens: int = 2048,
                 short_term_window: int = 5,
                 long_term_k: int = 3):
        """
        Args:
            max_context_tokens: Maximum number of tokens for context
            short_term_window: Number of recent messages from short-term
            long_term_k: Number of relevant messages from long-term
        """
        self.max_context_tokens = max_context_tokens
        self.short_term_window = short_term_window
        self.long_term_k = long_term_k
        
        # Initialize memory components
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

        logger.info("ConversationManager initialized")
    
    def get_enriched_context(self, 
                           session_id: str, 
                           current_query: str,
                           user_id: Optional[str] = None,
                           include_long_term: bool = True) -> Dict[str, any]:
        """
        Creates enriched context from short-term and long-term memory.
        
        Args:
            session_id: Current session ID
            current_query: Current user query
            user_id: User ID (to filter long-term memory)
            include_long_term: Whether to use long-term memory
            
        Returns:
            Dict containing enriched context and metadata
        """
        try:
            # 1. Get short-term context (recent conversation in session)
            short_context = self.short_term.get_recent_history(
                session_id=session_id, 
                window_size=self.short_term_window
            )
            
            # 2. Get long-term context (semantic search from history)
            long_context = []
            if include_long_term:
                long_context = self.long_term.retrieve_relevant_history(
                    query=current_query,
                    top_k=self.long_term_k,
                    user_id=user_id
                )
            
            # 3. Merge and format context
            merged_context = self._merge_contexts(
                current_query=current_query,
                short_context=short_context,
                long_context=long_context
            )
            
            # 4. Check and truncate context if too long
            final_context = self._truncate_context_if_needed(merged_context)
            
            return {
                "enriched_query": final_context["enriched_query"],
                "conversation_history": final_context["conversation_history"],
                "relevant_background": final_context["relevant_background"],
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "short_term_messages": len(short_context),
                    "long_term_messages": len(long_context),
                    "context_length": len(final_context["enriched_query"]),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get enriched context: {e}")
            # Fallback: return original query
            return {
                "enriched_query": current_query,
                "conversation_history": "",
                "relevant_background": "",
                "metadata": {"error": str(e)}
            }
    
    def _merge_contexts(self, 
                       current_query: str,
                       short_context: List[Dict],
                       long_context: List[Dict]) -> Dict[str, str]:
        """
        Merges contexts from short-term and long-term into a suitable format for LLM.
        """
        # Format short-term context (recent conversation)
        conversation_history = ""
        if short_context:
            history_parts = []
            for msg in short_context:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role_label}: {msg['message']}")
            conversation_history = "\n".join(history_parts)
        
        # Format long-term context (relevant background information)
        relevant_background = ""
        if long_context:
            background_parts = []
            for msg in long_context:
                # Add context from history with score
                score = msg.get("score", 0)
                timestamp = msg.get("timestamp", "")
                role_label = "User" if msg["role"] == "user" else "Assistant"
                
                # Format: [timestamp] Role: message (relevance: score)
                background_parts.append(
                    f"[{timestamp[:10]}] {role_label}: {msg['message']} (relevance: {score:.3f})"
                )
            relevant_background = "\n".join(background_parts)
        
        # Create enriched query
        enriched_parts = [current_query]
        
        if conversation_history:
            enriched_parts.insert(0, f"Recent conversation:\n{conversation_history}\n")
        
        if relevant_background:
            enriched_parts.insert(-1, f"Relevant background:\n{relevant_background}\n")
        
        enriched_query = "\n".join(enriched_parts)
        
        return {
            "enriched_query": enriched_query,
            "conversation_history": conversation_history,
            "relevant_background": relevant_background
        }
    
    def _truncate_context_if_needed(self, context: Dict[str, str]) -> Dict[str, str]:
        """
        Truncates context if it exceeds the token limit.
        Prioritizes: current query > short-term > long-term
        """
        enriched_query = context["enriched_query"]
        
        # Estimate tokens (4 chars ≈ 1 token)
        estimated_tokens = len(enriched_query) // 4
        
        if estimated_tokens <= self.max_context_tokens:
            return context
        
        logger.warning(f"Context too long ({estimated_tokens} tokens), truncating...")
        
        # Prioritize truncation
        conversation_history = context["conversation_history"]
        relevant_background = context["relevant_background"]
        
        # Keep at least current query
        current_query_lines = enriched_query.split("\n")
        current_query = current_query_lines[-1]  # Original query at the end
        
        # Truncate background first
        if relevant_background:
            background_lines = relevant_background.split("\n")
            # Keep at most 2 background messages
            truncated_background = "\n".join(background_lines[:2])
            relevant_background = truncated_background
        
        # Truncate conversation history if still too long
        if conversation_history:
            history_lines = conversation_history.split("\n")
            # Keep at most 3 recent conversations
            truncated_history = "\n".join(history_lines[-6:])  # 3 turns = 6 lines
            conversation_history = truncated_history
        
        # Reconstruct enriched query
        new_parts = [current_query]
        if conversation_history:
            new_parts.insert(0, f"Recent conversation:\n{conversation_history}\n")
        if relevant_background:
            new_parts.insert(-1, f"Relevant background:\n{relevant_background}\n")
        
        return {
            "enriched_query": "\n".join(new_parts),
            "conversation_history": conversation_history,
            "relevant_background": relevant_background
        }
    
    def update_memory(self, 
                     session_id: str, 
                     message: str, 
                     role: str,
                     user_id: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> None:
        """
        Updates both short-term and long-term memory with the new message.
        
        Args:
            session_id: Session ID
            message: Message content
            role: "user" or "assistant"
            user_id: User ID
            metadata: Additional information
        """
        try:
            # Update short-term memory
            self.short_term.store_message(
                session_id=session_id,
                message=message,
                role=role,
                metadata=metadata
            )
            
            # Update long-term memory
            self.long_term.store_message(
                session_id=session_id,
                message=message,
                role=role,
                user_id=user_id,
                metadata=metadata
            )
            
            logger.debug(f"Updated memory for session {session_id}, role: {role}")
            
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise
    
    def get_session_summary(self, session_id: str) -> Dict:
        """
        Gets a summary of the current session.
        """
        try:
            short_term_info = self.short_term.get_session_info(session_id)
            
            return {
                "session_id": session_id,
                "short_term": short_term_info,
                "status": "active" if short_term_info.get("exists") else "inactive"
            }
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {"session_id": session_id, "error": str(e)}
    
    def get_user_summary(self, user_id: str) -> Dict:
        """
        Gets a summary of the user from long-term memory.
        """
        try:
            return self.long_term.get_user_context_summary(user_id)
        except Exception as e:
            logger.error(f"Failed to get user summary: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    def clear_session(self, session_id: str) -> None:
        """
        Clears the session from short-term memory.
        Long-term memory remains to serve future sessions.
        """
        try:
            self.short_term.clear_session(session_id)
            logger.info(f"Cleared session {session_id} from short-term memory")
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            raise
    
    def get_memory_stats(self) -> Dict:
        """
        Gets overall statistics about the memory system.
        """
        try:
            short_term_stats = self.short_term.get_stats()
            long_term_stats = self.long_term.get_stats()
            
            return {
                "short_term": short_term_stats,
                "long_term": long_term_stats,
                "configuration": {
                    "max_context_tokens": self.max_context_tokens,
                    "short_term_window": self.short_term_window,
                    "long_term_k": self.long_term_k
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}