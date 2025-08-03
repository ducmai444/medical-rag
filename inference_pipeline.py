import pandas as pd
from evaluation import evaluate_llm
from llm_components.prompt_templates import InferenceTemplate
from monitoring import PromptMonitoringManager
from rag.retriever import VectorRetriever
from settings import settings
from typing import Optional, Dict
import uuid
import logging
import time

from openai import OpenAI

# Import conversation manager (with error handling)
try:
    from memory.conversation_manager import ConversationManager
    CONVERSATION_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ConversationManager not available: {e}")
    CONVERSATION_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


### llm deploy openai compatible server
# client = OpenAI(
#     api_key='YOUR_API_KEY',
#     base_url="http://0.0.0.0:23333/v1"
# )
# model_name = client.models.list().data[0].id
# response = client.chat.completions.create(
#   model=model_name,
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": " provide three suggestions about time management"},
#   ],
#     temperature=0.8,
#     top_p=0.8
# )

class LLM_RAG:
    def __init__(self) -> None:
        self.llm_client = OpenAI(
            api_key=settings.LLMDEPLOY_API_KEY,
            )
        self.template = InferenceTemplate()
        self.prompt_monitoring_manager = PromptMonitoringManager()

    def generate(
        self,
        query: str,
        enable_rag: bool = False,
        enable_evaluation: bool = False,
        enable_monitoring: bool = True,
    ) -> dict:
        prompt_template = self.template.create_template(enable_rag=enable_rag)
        prompt_template_variables = {
            "question": query,
        }

        if enable_rag is True:
            retriever = VectorRetriever(query=query)
            hits = retriever.retrieve_top_k(
                k=settings.TOP_K, to_expand_to_n_queries=settings.EXPAND_N_QUERY
            )
            context = retriever.rerank(hits=hits, keep_top_k=settings.KEEP_TOP_K)
            prompt_template_variables["context"] = context

            prompt = prompt_template.format(question=query, context=context)
        else:
            prompt = prompt_template.format(question=query)

        # model_name = self.llm_client.models.list().data[0].id
        response = self.llm_client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."}, # change system prompt
                {"role": "user", "content": prompt},
            ]
        )
        answer = response.choices[0].message.content

        if enable_evaluation is True:
            evaluation_result = evaluate_llm(query=query, output=answer)
        else:
            evaluation_result = None

        if enable_monitoring is True:
            if evaluation_result is not None:
                metadata = {"llm_evaluation_result": evaluation_result}
            else:
                metadata = None

            self.prompt_monitoring_manager.log(
                prompt=prompt,
                prompt_template=prompt_template.template,
                prompt_template_variables=prompt_template_variables,
                output=answer,
                metadata=metadata,
            )
            self.prompt_monitoring_manager.log_chain(
                query=query, response=answer, eval_output=evaluation_result
            )

        return {"answer": answer, "llm_evaluation_result": evaluation_result}


class ConversationalLLM_RAG:
    """
    Extended RAG Pipeline với conversation memory support.
    Wrapper around LLM_RAG để thêm conversation capabilities.
    """
    
    def __init__(self, 
                 max_context_tokens: int = 2048,
                 short_term_window: int = 5,
                 long_term_k: int = 3):
        """
        Args:
            max_context_tokens: Giới hạn token tối đa cho context
            short_term_window: Số lượng message gần nhất từ short-term
            long_term_k: Số lượng message liên quan từ long-term
        """
        # Khởi tạo RAG pipeline gốc
        self.rag_pipeline = LLM_RAG()
        
        # Khởi tạo conversation manager nếu có thể
        if CONVERSATION_MANAGER_AVAILABLE:
            try:
                self.conversation_manager = ConversationManager(
                    max_context_tokens=max_context_tokens,
                    short_term_window=short_term_window,
                    long_term_k=long_term_k
                )
                self.conversation_enabled = True
                logger.info("ConversationalLLM_RAG initialized with conversation support")
            except Exception as e:
                logger.error(f"Failed to initialize ConversationManager: {e}")
                self.conversation_enabled = False
        else:
            self.conversation_enabled = False
            logger.warning("ConversationalLLM_RAG initialized without conversation support")
    
    def generate(self,
                query: str,
                session_id: Optional[str] = None,
                user_id: Optional[str] = None,
                enable_rag: bool = True,
                enable_conversation: bool = True,
                enable_evaluation: bool = False,
                enable_monitoring: bool = True) -> Dict:
        """
        Generate response với conversation context support.
        
        Args:
            query: Câu hỏi của user
            session_id: ID session (tự tạo nếu None)
            user_id: ID user (để filter long-term memory)
            enable_rag: Có sử dụng RAG không
            enable_conversation: Có sử dụng conversation context không
            enable_evaluation: Có đánh giá không
            enable_monitoring: Có monitoring không
            
        Returns:
            Dict chứa answer và metadata
        """
        # Tạo session_id nếu chưa có
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Nếu conversation không enabled, fallback về RAG pipeline gốc
        if not self.conversation_enabled or not enable_conversation:
            return self._generate_without_conversation(
                query=query,
                enable_rag=enable_rag,
                enable_evaluation=enable_evaluation,
                enable_monitoring=enable_monitoring,
                session_id=session_id
            )
        
        try:
            # 1. Lấy enriched context từ conversation memory
            context_data = self.conversation_manager.get_enriched_context(
                session_id=session_id,
                current_query=query,
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
            try:
                self.conversation_manager.update_memory(
                    session_id=session_id,
                    message=query,
                    role="user",
                    user_id=user_id
                )
                
                self.conversation_manager.update_memory(
                    session_id=session_id,
                    message=rag_result["answer"],
                    role="assistant",
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"Failed to update conversation memory: {e}")
            
            # 5. Trả về kết quả với context metadata
            return {
                "answer": rag_result["answer"],
                "session_id": session_id,
                "conversation_enabled": True,
                "context_metadata": context_data.get("metadata", {}),
                "conversation_history": context_data.get("conversation_history", ""),
                "relevant_background": context_data.get("relevant_background", ""),
                "llm_evaluation_result": rag_result.get("llm_evaluation_result")
            }
            
        except Exception as e:
            logger.error(f"Error in conversational RAG: {e}")
            # Fallback: sử dụng RAG pipeline gốc nếu có lỗi
            return self._generate_without_conversation(
                query=query,
                enable_rag=enable_rag,
                enable_evaluation=enable_evaluation,
                enable_monitoring=enable_monitoring,
                session_id=session_id,
                error=str(e)
            )
    
    def _generate_without_conversation(self,
                                     query: str,
                                     enable_rag: bool,
                                     enable_evaluation: bool,
                                     enable_monitoring: bool,
                                     session_id: str,
                                     error: Optional[str] = None) -> Dict:
        """
        Fallback method sử dụng RAG pipeline gốc.
        """
        try:
            rag_result = self.rag_pipeline.generate(
                query=query,
                enable_rag=enable_rag,
                enable_evaluation=enable_evaluation,
                enable_monitoring=enable_monitoring
            )
            
            result = {
                "answer": rag_result["answer"],
                "session_id": session_id,
                "conversation_enabled": False,
                "llm_evaluation_result": rag_result.get("llm_evaluation_result")
            }
            
            if error:
                result["conversation_error"] = error
                result["fallback_used"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fallback RAG pipeline: {e}")
            return {
                "answer": "Sorry, I encountered an error processing your request.",
                "session_id": session_id,
                "conversation_enabled": False,
                "error": str(e)
            }
    
    def get_session_info(self, session_id: str) -> Dict:
        """Lấy thông tin về session."""
        if self.conversation_enabled:
            try:
                return self.conversation_manager.get_session_summary(session_id)
            except Exception as e:
                logger.error(f"Failed to get session info: {e}")
                return {"session_id": session_id, "error": str(e)}
        else:
            return {"session_id": session_id, "conversation_enabled": False}
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Lấy profile user từ long-term memory."""
        if self.conversation_enabled:
            try:
                return self.conversation_manager.get_user_summary(user_id)
            except Exception as e:
                logger.error(f"Failed to get user profile: {e}")
                return {"user_id": user_id, "error": str(e)}
        else:
            return {"user_id": user_id, "conversation_enabled": False}
    
    def clear_session(self, session_id: str) -> None:
        """Xóa session khỏi short-term memory."""
        if self.conversation_enabled:
            try:
                self.conversation_manager.clear_session(session_id)
                logger.info(f"Cleared session {session_id}")
            except Exception as e:
                logger.error(f"Failed to clear session: {e}")
        else:
            logger.warning("Conversation not enabled, cannot clear session")
    
    def get_memory_stats(self) -> Dict:
        """Lấy thống kê memory system."""
        if self.conversation_enabled:
            try:
                return self.conversation_manager.get_memory_stats()
            except Exception as e:
                logger.error(f"Failed to get memory stats: {e}")
                return {"error": str(e)}
        else:
            return {"conversation_enabled": False}


# Backward compatibility: alias cho class gốc
LLMRag = LLM_RAG  # Để tương thích với code cũ

# Medical imports
try:
    from others.medical_pipeline import MedicalRAGPipeline, MedicalContext
    MEDICAL_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Medical pipeline not available: {e}")
    MEDICAL_PIPELINE_AVAILABLE = False
    MedicalRAGPipeline = None
    MedicalContext = None

class MedicalLLM_RAG:
    """
    Medical-enhanced RAG pipeline that integrates Vietnamese translation,
    medical NER, UMLS knowledge, and standard RAG retrieval.
    """
    
    def __init__(self, 
                 umls_api_key: str = None, 
                 enable_translation: bool = True,
                 enable_caching: bool = False,
                 max_workers: int = 4):
        """
        Initialize Medical RAG pipeline.
        
        Args:
            umls_api_key: API key for UMLS access
            enable_translation: Enable Vietnamese translation
            enable_caching: Enable Redis caching
            max_workers: Max workers for parallel processing
        """
        # Initialize standard RAG pipeline
        self.rag_pipeline = LLM_RAG()
        
        # Initialize medical pipeline
        self.medical_enabled = MEDICAL_PIPELINE_AVAILABLE and umls_api_key
        if self.medical_enabled:
            try:
                self.medical_pipeline = MedicalRAGPipeline(
                    umls_api_key=umls_api_key,
                    enable_translation=enable_translation,
                    enable_caching=enable_caching,
                    max_workers=max_workers
                )
                logger.info("✅ Medical RAG pipeline initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize medical pipeline: {e}")
                self.medical_pipeline = None
                self.medical_enabled = False
        else:
            self.medical_pipeline = None
            logger.warning("Medical pipeline disabled (missing dependencies or API key)")
    
    def generate(self, 
                 query: str, 
                 enable_rag: bool = True,
                 enable_medical: bool = True,
                 medical_priority_weight: float = 0.7,
                 enable_evaluation: bool = False,
                 enable_monitoring: bool = True,
                 **kwargs) -> dict:
        """
        Generate response using medical-enhanced RAG.
        
        Args:
            query: User query
            enable_rag: Enable standard RAG retrieval
            enable_medical: Enable medical knowledge retrieval
            medical_priority_weight: Weight for medical context vs document context
            enable_evaluation: Enable evaluation
            enable_monitoring: Enable monitoring
            **kwargs: Additional arguments for RAG pipeline
            
        Returns:
            Dict containing answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Check if query is medical-related and process medical context
            is_medical_query = False
            medical_context = None
            query_for_rag = query
            
            if enable_medical and self.medical_enabled:
                is_medical_query = self.medical_pipeline.is_medical_query(query)
                
                if is_medical_query:
                    logger.info("Processing medical query")
                    medical_context = self.medical_pipeline.process_query(query)
                    
                    # Use translated query for RAG if available
                    if medical_context.translated_query != medical_context.original_query:
                        query_for_rag = medical_context.translated_query
                        logger.info(f"Using translated query for RAG: {query_for_rag}")
            
            # Step 2: Standard RAG retrieval
            rag_context = ""
            rag_result = None
            if enable_rag:
                try:
                    retriever = VectorRetriever(query=query_for_rag)
                    hits = retriever.retrieve_top_k(
                        k=settings.TOP_K, to_expand_to_n_queries=settings.EXPAND_N_QUERY
                    )
                    rag_context = retriever.rerank(hits=hits, keep_top_k=settings.KEEP_TOP_K)
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
                    rag_context = ""
            
            # Step 3: Merge medical context with RAG context
            final_context = self._merge_contexts(
                medical_context=medical_context,
                rag_context=rag_context,
                medical_weight=medical_priority_weight
            )
            
            # Step 4: Generate final response using LLM
            final_prompt = self._create_enhanced_prompt(
                original_query=query,
                context=final_context,
                medical_context=medical_context
            )
            
            # Generate LLM response
            response = self.rag_pipeline.llm_client.chat.completions.create(
                model="o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant. Provide accurate and comprehensive answers based on the provided context."},
                    {"role": "user", "content": final_prompt},
                ]
            )
            answer = response.choices[0].message.content
            
            # Step 5: Prepare result with medical metadata
            result = {
                "answer": answer,
                "is_medical_query": is_medical_query,
                "pipeline_type": "medical_rag" if is_medical_query else "standard_rag",
                "processing_time": time.time() - start_time,
                "context": final_context
            }
            
            # Add medical metadata if available
            if medical_context:
                result["medical_metadata"] = self.medical_pipeline.get_medical_metadata(medical_context)
                result["medical_confidence"] = medical_context.confidence_score
                result["translation_used"] = medical_context.original_query != medical_context.translated_query
            
            # Handle evaluation and monitoring
            if enable_evaluation:
                try:
                    evaluation_result = evaluate_llm(query=query, output=answer)
                    result["llm_evaluation_result"] = evaluation_result
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                    result["llm_evaluation_result"] = None
            
            if enable_monitoring:
                try:
                    metadata = {"medical_enhanced": is_medical_query}
                    if result.get("llm_evaluation_result"):
                        metadata["llm_evaluation_result"] = result["llm_evaluation_result"]
                    
                    self.rag_pipeline.prompt_monitoring_manager.log(
                        prompt=final_prompt,
                        prompt_template="medical_enhanced_template",
                        prompt_template_variables={"query": query, "context": final_context},
                        output=answer,
                        metadata=metadata
                    )
                    self.rag_pipeline.prompt_monitoring_manager.log_chain(
                        query=query, 
                        response=answer, 
                        eval_output=result.get("llm_evaluation_result")
                    )
                except Exception as e:
                    logger.warning(f"Monitoring failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Medical RAG generation failed: {e}")
            # Fallback to standard RAG
            if enable_rag:
                try:
                    fallback_result = self.rag_pipeline.generate(
                        query=query, 
                        enable_rag=True,
                        enable_evaluation=enable_evaluation,
                        enable_monitoring=enable_monitoring
                    )
                    fallback_result["pipeline_type"] = "fallback_standard_rag"
                    fallback_result["medical_error"] = str(e)
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            return {
                "answer": "I apologize, but I encountered an error processing your request.",
                "pipeline_type": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _merge_contexts(self, 
                       medical_context: Optional[MedicalContext],
                       rag_context: str,
                       medical_weight: float = 0.7) -> str:
        """
        Merge medical context with RAG context using weighted priority.
        
        Args:
            medical_context: Medical knowledge context
            rag_context: Context from standard RAG retrieval
            medical_weight: Priority weight for medical context (0.0-1.0)
            
        Returns:
            Merged context string
        """
        context_parts = []
        
        # Add medical context with high priority
        if medical_context and medical_context.umls_results:
            medical_text = self.medical_pipeline.format_medical_context(medical_context)
            if medical_text:
                context_parts.append(f"MEDICAL KNOWLEDGE:\n{medical_text}")
        
        # Add RAG context
        if rag_context and rag_context.strip():
            context_parts.append(f"DOCUMENT CONTEXT:\n{rag_context}")
        
        # Prioritize based on weight
        if len(context_parts) == 2:
            if medical_weight > 0.5:
                # Medical context first
                return f"{context_parts[0]}\n\n{context_parts[1]}"
            else:
                # Document context first
                return f"{context_parts[1]}\n\n{context_parts[0]}"
        
        return "\n\n".join(context_parts)
    
    def _create_enhanced_prompt(self, 
                              original_query: str,
                              context: str,
                              medical_context: Optional[MedicalContext]) -> str:
        """
        Create an enhanced prompt that incorporates medical knowledge.
        
        Args:
            original_query: Original user query
            context: Merged context from medical and RAG sources
            medical_context: Medical knowledge context
            
        Returns:
            Enhanced prompt string
        """
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context Information:\n{context}")
        
        # Add medical entities information if available
        if medical_context and medical_context.medical_terms:
            medical_terms_str = ", ".join(medical_context.medical_terms)
            prompt_parts.append(f"Medical Terms Identified: {medical_terms_str}")
        
        # Add final relations if available
        if medical_context and medical_context.final_relations:
            relations_str = []
            for subject, relation, obj in medical_context.final_relations[:5]:  # Top 5 relations
                relations_str.append(f"({subject}, {relation}, {obj})")
            if relations_str:
                prompt_parts.append(f"Medical Relations: {'; '.join(relations_str)}")
        
        prompt_parts.append(f"Question: {original_query}")
        
        if medical_context and medical_context.umls_results:
            prompt_parts.append(
                "Please provide a comprehensive medical answer that integrates both "
                "the medical knowledge and any relevant document information above. "
                "Prioritize medical accuracy and be specific about medical terms and relationships."
            )
        else:
            prompt_parts.append(
                "Please provide a comprehensive answer based on the context information above."
            )
        
        return "\n\n".join(prompt_parts)
    
    def get_medical_stats(self) -> dict:
        """Get medical pipeline statistics."""
        if self.medical_enabled and self.medical_pipeline:
            return self.medical_pipeline.get_stats()
        return {"medical_enabled": False}
    
    def is_medical_enabled(self) -> bool:
        """Check if medical pipeline is enabled."""
        return self.medical_enabled