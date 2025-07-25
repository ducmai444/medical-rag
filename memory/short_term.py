from typing import List, Dict, Optional
from collections import defaultdict, deque
import time
import threading
from datetime import datetime, timedelta

class ShortTermMemory:
    """
    Manages short-term memory for recent conversations in the current session.
    Uses in-memory storage with TTL and message limit.
    """
    
    def __init__(self, max_messages_per_session: int = 20, ttl_hours: int = 24):
        """
        Args:
            max_messages_per_session: Maximum number of messages to store per session
            ttl_hours: Session lifetime (hours)
        """
        self.max_messages_per_session = max_messages_per_session
        self.ttl_hours = ttl_hours
        
        # Store messages by session_id
        # Format: {session_id: deque([{message, role, timestamp}, ...])}
        self._sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_messages_per_session))
        
        # Store session timestamps for cleanup
        self._session_timestamps: Dict[str, datetime] = {}
        
        # Lock for thread-safe
        self._lock = threading.RLock()
    
    def store_message(self, session_id: str, message: str, role: str, metadata: Optional[Dict] = None) -> None:
        """
        Stores a message in short-term memory.
        
        Args:
            session_id: ID of the session
            message: Message content
            role: "user" or "assistant"
            metadata: Additional information (optional)
        """
        with self._lock:
            timestamp = datetime.now()
            
            message_data = {
                "message": message,
                "role": role,
                "timestamp": timestamp,
                "metadata": metadata or {}
            }
            
            self._sessions[session_id].append(message_data)
            self._session_timestamps[session_id] = timestamp
            
            # Cleanup expired sessions
            self._cleanup_expired_sessions()
    
    def get_recent_history(self, session_id: str, window_size: int = 5) -> List[Dict]:
        """
        Retrieves recent conversation history for a session.
        
        Args:
            session_id: ID of the session
            window_size: Number of most recent messages to retrieve
            
        Returns:
            List of the most recent messages, sorted by time (oldest to newest)
        """
        with self._lock:
            if session_id not in self._sessions:
                return []
            
            messages = list(self._sessions[session_id])
            
            # Get window_size most recent messages
            recent_messages = messages[-window_size:] if len(messages) > window_size else messages
            
            return recent_messages
    
    def get_conversation_context(self, session_id: str, window_size: int = 5) -> str:
        """
        Creates a context string from recent conversation history.
        
        Args:
            session_id: ID of the session
            window_size: Number of most recent messages
            
        Returns:
            Formatted context string for LLM
        """
        recent_history = self.get_recent_history(session_id, window_size)
        
        if not recent_history:
            return ""
        
        context_parts = []
        for msg in recent_history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg['message']}")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str) -> None:
        """Clears all history for a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            if session_id in self._session_timestamps:
                del self._session_timestamps[session_id]
    
    def get_session_info(self, session_id: str) -> Dict:
        """
        Retrieves general information about a session.
        
        Returns:
            Dict containing session information (number of messages, creation time, etc.)
        """
        with self._lock:
            if session_id not in self._sessions:
                return {"exists": False}
            
            messages = self._sessions[session_id]
            return {
                "exists": True,
                "message_count": len(messages),
                "created_at": self._session_timestamps.get(session_id),
                "last_activity": messages[-1]["timestamp"] if messages else None
            }
    
    def _cleanup_expired_sessions(self) -> None:
        """Clears expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, timestamp in self._session_timestamps.items():
            if current_time - timestamp > timedelta(hours=self.ttl_hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.clear_session(session_id)
    
    def get_active_sessions(self) -> List[str]:
        """Retrieves a list of active sessions."""
        with self._lock:
            self._cleanup_expired_sessions()
            return list(self._sessions.keys())
    
    def get_stats(self) -> Dict:
        """Retrieves general statistics about short-term memory."""
        with self._lock:
            total_messages = sum(len(session) for session in self._sessions.values())
            return {
                "active_sessions": len(self._sessions),
                "total_messages": total_messages,
                "max_messages_per_session": self.max_messages_per_session,
                "ttl_hours": self.ttl_hours
            }
