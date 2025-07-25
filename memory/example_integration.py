"""
Example integration của Memory system với Inference Pipeline.
Minh họa cách sử dụng ConversationManager trong thực tế.
"""

import uuid
from memory import ConversationManager
from inference_pipeline import LLM_RAG

class ConversationalRAGPipeline:
    """
    Extended RAG Pipeline với conversation memory support.
    Wrapper around LLM_RAG để thêm conversation capabilities.
    """
    
    def __init__(self):
        self.rag_pipeline = LLM_RAG()
        self.conversation_manager = ConversationManager(
            max_context_tokens=2048,
            short_term_window=5,
            long_term_k=3
        )
    
    def chat(self, 
             user_query: str,
             session_id: str = None,
             user_id: str = None,
             enable_rag: bool = True,
             enable_evaluation: bool = False,
             enable_monitoring: bool = True) -> dict:
        """
        Xử lý câu hỏi với conversation context.
        
        Args:
            user_query: Câu hỏi của user
            session_id: ID session (tự tạo nếu None)
            user_id: ID user (để filter long-term memory)
            enable_rag: Có sử dụng RAG không
            enable_evaluation: Có đánh giá không
            enable_monitoring: Có monitoring không
            
        Returns:
            Dict chứa answer và metadata
        """
        # Tạo session_id nếu chưa có
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        try:
            # 1. Lấy enriched context từ conversation memory
            context_data = self.conversation_manager.get_enriched_context(
                session_id=session_id,
                current_query=user_query,
                user_id=user_id,
                include_long_term=enable_rag
            )
            
            # 2. Sử dụng enriched query cho RAG pipeline
            enriched_query = context_data["enriched_query"]
            
            # 3. Gọi RAG pipeline với enriched query
            rag_result = self.rag_pipeline.generate(
                query=enriched_query,
                enable_rag=enable_rag,
                enable_evaluation=enable_evaluation,
                enable_monitoring=enable_monitoring
            )
            
            # 4. Lưu user query và bot response vào memory
            self.conversation_manager.update_memory(
                session_id=session_id,
                message=user_query,
                role="user",
                user_id=user_id
            )
            
            self.conversation_manager.update_memory(
                session_id=session_id,
                message=rag_result["answer"],
                role="assistant",
                user_id=user_id
            )
            
            # 5. Trả về kết quả với context metadata
            return {
                "answer": rag_result["answer"],
                "session_id": session_id,
                "context_metadata": context_data["metadata"],
                "conversation_history": context_data["conversation_history"],
                "relevant_background": context_data["relevant_background"],
                "llm_evaluation_result": rag_result.get("llm_evaluation_result")
            }
            
        except Exception as e:
            # Fallback: sử dụng RAG pipeline gốc nếu có lỗi
            print(f"Error in conversational RAG: {e}")
            rag_result = self.rag_pipeline.generate(
                query=user_query,
                enable_rag=enable_rag,
                enable_evaluation=enable_evaluation,
                enable_monitoring=enable_monitoring
            )
            return {
                "answer": rag_result["answer"],
                "session_id": session_id,
                "error": str(e),
                "fallback": True
            }
    
    def get_session_info(self, session_id: str) -> dict:
        """Lấy thông tin về session."""
        return self.conversation_manager.get_session_summary(session_id)
    
    def get_user_profile(self, user_id: str) -> dict:
        """Lấy profile user từ long-term memory."""
        return self.conversation_manager.get_user_summary(user_id)
    
    def clear_session(self, session_id: str) -> None:
        """Xóa session khỏi short-term memory."""
        self.conversation_manager.clear_session(session_id)
    
    def get_memory_stats(self) -> dict:
        """Lấy thống kê memory system."""
        return self.conversation_manager.get_memory_stats()


# Example usage
if __name__ == "__main__":
    # Khởi tạo conversational RAG pipeline
    conv_rag = ConversationalRAGPipeline()
    
    # Simulate conversation
    session_id = "test_session_001"
    user_id = "user_123"
    
    # Lượt 1
    print("=== Lượt 1 ===")
    result1 = conv_rag.chat(
        user_query="Tôi bị đau đầu, nên làm gì?",
        session_id=session_id,
        user_id=user_id
    )
    print(f"Bot: {result1['answer']}")
    
    # Lượt 2 - câu hỏi phụ thuộc vào context
    print("\n=== Lượt 2 ===")
    result2 = conv_rag.chat(
        user_query="Còn về lịch sử bệnh của tôi thì sao?",
        session_id=session_id,
        user_id=user_id
    )
    print(f"Bot: {result2['answer']}")
    print(f"Context used: {result2['conversation_history']}")
    
    # Lượt 3 - tiếp tục conversation
    print("\n=== Lượt 3 ===")
    result3 = conv_rag.chat(
        user_query="Tôi có nên đi khám bác sĩ không?",
        session_id=session_id,
        user_id=user_id
    )
    print(f"Bot: {result3['answer']}")
    
    # Kiểm tra memory stats
    print("\n=== Memory Stats ===")
    stats = conv_rag.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    # Kiểm tra user profile
    print("\n=== User Profile ===")
    profile = conv_rag.get_user_profile(user_id)
    print(f"User profile: {profile}") 