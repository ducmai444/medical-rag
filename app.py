import streamlit as st
from inference_pipeline import ConversationalLLM_RAG, LLM_RAG
import wandb
from settings import settings
from vector_db.qdrant import QdrantDatabaseConnector
from sentence_transformers import SentenceTransformer
from qdrant_client.http.exceptions import UnexpectedResponse
import pypdf
import pandas as pd
import docx
import io
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

wandb.login(key=settings.WANDB_API_KEY)

def extract_text_from_file(uploaded_file):
    """
    Extract text from uploaded file (PDF, CSV, DOCX).
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Extracted text
    """
    file_type = uploaded_file.type
    text = ""

    if file_type == "application/pdf":
        pdf_reader = pypdf.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    
    elif file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        df = pd.read_csv(uploaded_file) if file_type == "text/csv" else pd.read_excel(uploaded_file)
        text = df.to_string()
    
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    
    return text

def init_qdrant_collection(reset=True):
    """Khởi tạo mới collection vector_documents bằng cách xóa và tạo lại."""
    qdrant_client = QdrantDatabaseConnector()
    collection_name = "vector_documents"

    try:
        try:
            qdrant_client._instance.delete_collection(collection_name)
            logger.info(f"Deleted existing collection {collection_name}")
        except Exception as e:
            logger.info(f"Collection {collection_name} doesn't exist or couldn't be deleted: {str(e)}")
        
        # Create new collection with index
        qdrant_client.create_vector_collection(collection_name)
        logger.info(f"Created new collection {collection_name} with vector size {settings.EMBEDDING_SIZE}")
    except UnexpectedResponse as e:
        if "already exists" in str(e):
            logger.info(f"Collection {collection_name} already exists, skipping creation.")
        else:
            logger.exception(f"Failed to create collection {collection_name}: {str(e)}")
            raise
    except Exception as e:
        logger.exception(f"Failed to create collection {collection_name}: {str(e)}")
        raise

def initialize_session_state():
    """Create new session state for Streamlit."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'use_conversation' not in st.session_state:
        st.session_state.use_conversation = True

def main():
    init_qdrant_collection()
    initialize_session_state()
    
    st.title("🤖 Advanced RAG with Conversation Memory")
    
    # Sidebar cho configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select RAG Model",
            ["Standard RAG", "Conversational RAG", "Medical RAG"],
            help="Choose the type of RAG model to use"
        )
        
        # Medical RAG configuration
        if model_type == "Medical RAG":
            st.subheader("🏥 Medical RAG Settings")
            
            umls_api_key = st.text_input(
                "UMLS API Key",
                type="password",
                help="Enter your UMLS API key for medical knowledge retrieval"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                enable_translation = st.checkbox("Enable Vietnamese Translation", value=True)
                enable_medical_ner = st.checkbox("Enable Medical NER", value=True)
            
            with col2:
                enable_umls = st.checkbox("Enable UMLS Knowledge", value=True)
                medical_priority = st.slider("Medical Knowledge Priority", 0.0, 1.0, 0.7, 0.1)
            
            max_umls_results = st.slider("Max UMLS Results", 1, 5, 3)
            
            # Advanced medical settings
            with st.expander("🔧 Advanced Medical Settings"):
                enable_caching = st.checkbox("Enable Redis Caching", value=False)
                max_workers = st.slider("Max Workers", 1, 8, 4)
        
        # Session info for conversational models
        if model_type in ["Conversational RAG", "Medical RAG"]:
            st.subheader("💬 Session Information")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 New Session"):
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.conversation_history = []
                    st.rerun()
            
            with col2:
                if st.button("📊 Session Stats"):
                    st.session_state.show_session_stats = not st.session_state.get('show_session_stats', False)
        
        # Settings
        st.subheader("🎛️ Settings")
        enable_rag = st.checkbox("Enable RAG", value=True)
        enable_conversation = st.checkbox("Enable Conversation Context", 
                                        value=True, 
                                        disabled=not st.session_state.use_conversation)
        enable_evaluation = st.checkbox("Enable Evaluation", value=False)
        enable_monitoring = st.checkbox("Enable Monitoring", value=True)
        
        # Chunking strategy selection
        st.subheader("📄 Chunking Strategy")
        chunking_strategy = st.selectbox(
            "Select chunking method:",
            ["intelligent", "semantic", "structural", "fast"],
            index=0,
            help="intelligent: Structure + Semantic aware\nsemantic: Sentence embedding based\nstructural: Paragraph aware\nfast: Token-based (fastest)"
        )

    # File upload section
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a file to update Vector DB (Qdrant)",
        type=["pdf", "csv", "xlsx", "docx"]
    )

    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            # Extract text and store in Qdrant
            text = extract_text_from_file(uploaded_file)
            if text:
                # Show chunking info
                with st.expander("📊 Chunking Information"):
                    from utils.chunking import chunk_text
                    
                    # Test chunking với strategy đã chọn
                    chunks = chunk_text(text, strategy=chunking_strategy)
                    st.write(f"**Strategy:** {chunking_strategy}")
                    st.write(f"**Total chunks:** {len(chunks)}")
                    st.write(f"**Text length:** {len(text):,} characters")
                    
                    # Show first few chunks
                    if len(chunks) > 0:
                        st.write("**Sample chunks:**")
                        for i, chunk in enumerate(chunks[:3]):
                            st.text(f"Chunk {i+1}: {chunk[:100]}...")
                
                qdrant_client = QdrantDatabaseConnector()
                metadata = {
                    "id": uploaded_file.name,
                    "author_id": "user_upload",  # Placeholder, can be dynamic
                    "source": uploaded_file.name,
                    "chunking_strategy": chunking_strategy
                }
                try:
                    # Create a custom chunking function that uses our strategy
                    def custom_chunk_text(text_input):
                        from utils.chunking import chunk_text as intelligent_chunk_text
                        return intelligent_chunk_text(text_input, strategy=chunking_strategy)
                    
                    # Temporarily replace the chunking function in the qdrant module
                    import utils.chunking
                    original_chunk_text = utils.chunking.chunk_text
                    utils.chunking.chunk_text = custom_chunk_text
                    
                    qdrant_client.store_text_with_chunking(
                        collection_name="vector_documents",
                        text=text,
                        metadata=metadata,
                    )
                    
                    # Restore original function
                    utils.chunking.chunk_text = original_chunk_text
                    
                    st.success(f"✅ File processed and stored in Qdrant using {chunking_strategy} chunking.")
                except Exception as e:
                    st.error(f"❌ Failed to store file in Qdrant: {str(e)}")
                    # Make sure to restore original function even if error occurs
                    try:
                        import utils.chunking
                        if 'original_chunk_text' in locals():
                            utils.chunking.chunk_text = original_chunk_text
                    except:
                        pass
            else:
                st.warning("⚠️ No text extracted from the file.")

    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("📋 Conversation History")
        for i, msg in enumerate(st.session_state.conversation_history[-5:]):  # Show last 5 messages
            with st.container():
                if msg["role"] == "user":
                    st.write(f"**👤 You:** {msg['content']}")
                else:
                    st.write(f"**🤖 Assistant:** {msg['content']}")
        st.divider()

    # Query input
    query = st.text_area("Enter your question:", placeholder="Type your question here...", height=100)

    # Submit button
    if st.button("🚀 Submit", type="primary"):
        if query:
            with st.spinner("Processing your question..."):
                try:
                    # Initialize model based on selection
                    if model_type == "Medical RAG":
                        if not umls_api_key:
                            st.error("⚠️ UMLS API key is required for Medical RAG")
                            st.stop()
                        
                        from inference_pipeline import MedicalLLM_RAG
                        
                        with st.spinner("🏥 Initializing Medical RAG..."):
                            model = MedicalLLM_RAG(
                                umls_api_key=umls_api_key,
                                enable_translation=enable_translation,
                                enable_caching=enable_caching,
                                max_workers=max_workers
                            )
                        
                        if not model.is_medical_enabled():
                            st.warning("⚠️ Medical pipeline could not be fully initialized. Some features may be limited.")
                        
                        st.success("✅ Medical RAG initialized successfully!")
                        
                        # Show medical stats
                        if st.session_state.get('show_session_stats', False):
                            with st.expander("🏥 Medical Pipeline Status", expanded=True):
                                medical_stats = model.get_medical_stats()
                                st.json(medical_stats)

                    elif model_type == "Conversational RAG":
                        from inference_pipeline import ConversationalLLM_RAG
                        
                        with st.spinner("💬 Initializing Conversational RAG..."):
                            model = ConversationalLLM_RAG()
                        
                        if not model.conversation_enabled:
                            st.warning("⚠️ Conversation memory could not be initialized. Falling back to standard RAG.")
                            fallback_reason = "Memory initialization failed"
                        else:
                            st.success("✅ Conversational RAG initialized successfully!")
                        
                        # Show session stats
                        if st.session_state.get('show_session_stats', False):
                            with st.expander("💬 Session Statistics", expanded=True):
                                if model.conversation_enabled:
                                    stats = model.get_memory_stats()
                                    st.json(stats)
                                else:
                                    st.info("Session statistics not available (memory disabled)")
                    
                    else:  # Standard RAG
                        from inference_pipeline import LLM_RAG
                        
                        with st.spinner("🔧 Initializing Standard RAG..."):
                            model = LLM_RAG()
                        
                        st.success("✅ Standard RAG initialized successfully!")
                        
                    # Generate response based on model type
                    if model_type == "Medical RAG":
                        result = model.generate(
                            query=query,
                            enable_rag=enable_rag,
                            enable_medical=enable_umls,
                            medical_priority_weight=medical_priority,
                            enable_evaluation=enable_evaluation,
                            enable_monitoring=enable_monitoring
                        )
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": query
                        })
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": result["answer"]
                        })
                        
                    elif model_type == "Conversational RAG":
                        result = model.generate(
                            query=query,
                            session_id=st.session_state.session_id,
                            user_id=st.session_state.user_id,
                            enable_rag=enable_rag,
                            enable_conversation=enable_conversation,
                            enable_evaluation=enable_evaluation,
                            enable_monitoring=enable_monitoring
                        )
                        
                        # Update conversation history from model
                        if result.get("conversation_history"):
                            # Don't overwrite, just add current exchange
                            st.session_state.conversation_history.append({
                                "role": "user",
                                "content": query
                            })
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": result["answer"]
                            })
                    
                    else:  # Standard RAG
                        result = model.generate(
                            query=query,
                            enable_rag=enable_rag,
                            enable_evaluation=enable_evaluation,
                            enable_monitoring=enable_monitoring
                        )
                        
                        # Add to conversation history for display
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": query
                        })
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": result["answer"]
                        })

                    # Display results
                    st.markdown("### 🤖 Assistant Response")
                    st.markdown(result["answer"])
                    
                    # Medical metadata (for Medical RAG)
                    if model_type == "Medical RAG" and result.get("medical_metadata"):
                        with st.expander("🏥 Medical Information", expanded=False):
                            medical_meta = result["medical_metadata"]
                            
                            # Medical terms detected
                            if medical_meta.get("medical_terms"):
                                st.write("**Medical Terms Detected:**")
                                for term in medical_meta["medical_terms"]:
                                    st.write(f"- {term}")
                            
                            # UMLS terms information
                            if medical_meta.get("umls_terms"):
                                st.write("**UMLS Knowledge Base Results:**")
                                for i, umls_term in enumerate(medical_meta["umls_terms"], 1):
                                    with st.container():
                                        st.write(f"**{i}. {umls_term['name']}** (CUI: {umls_term['cui']})")
                                        if umls_term.get("definition"):
                                            st.write(f"   - Definition: {umls_term['definition']}")
                                        st.write(f"   - Relations found: {umls_term['relations_count']}")
                                        st.write(f"   - Ranked relations: {umls_term['ranked_relations_count']}")
                            
                            # Final relations (medical knowledge graph)
                            if medical_meta.get("final_relations"):
                                st.write("**Medical Knowledge Relations:**")
                                for i, relation in enumerate(medical_meta["final_relations"][:10], 1):  # Top 10
                                    st.write(f"{i}. **{relation['subject']}** → _{relation['relation']}_ → **{relation['object']}**")
                            
                            # Processing metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                confidence = medical_meta.get("confidence_score", 0)
                                st.metric("Medical Confidence", f"{confidence:.2f}")
                            with col2:
                                processing_time = medical_meta.get("processing_time", 0)
                                st.metric("Processing Time", f"{processing_time:.2f}s")
                            with col3:
                                translation_status = "Yes" if medical_meta.get("translation_used") else "No"
                                st.metric("Translation Used", translation_status)
                    
                    # Conversation context (for Conversational RAG)
                    if model_type == "Conversational RAG" and result.get("conversation_history"):
                        with st.expander("💬 Conversation Context", expanded=False):
                            st.write("**Session Context Used:**")
                            if result.get("context_metadata"):
                                context_meta = result["context_metadata"]
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Short-term Messages", context_meta.get("short_term_messages", 0))
                                with col2:
                                    st.metric("Long-term Messages", context_meta.get("long_term_messages", 0))
                            
                            if result.get("relevant_background"):
                                st.write("**Relevant Background:**")
                                st.write(result["relevant_background"])
                    
                    # Standard metadata
                    with st.expander("📊 Response Metadata", expanded=False):
                        metadata_to_show = {
                            "Pipeline Type": result.get("pipeline_type", "unknown"),
                            "Processing Time": f"{result.get('processing_time', 0):.2f}s",
                            "RAG Enabled": enable_rag,
                        }
                        
                        if model_type == "Medical RAG":
                            metadata_to_show.update({
                                "Medical Query": result.get("is_medical_query", False),
                                "Medical Confidence": f"{result.get('medical_confidence', 0):.2f}",
                                "Translation Used": result.get("translation_used", False)
                            })
                            
                            if result.get("medical_metadata"):
                                umls_count = len(result["medical_metadata"].get("umls_terms", []))
                                relations_count = len(result["medical_metadata"].get("final_relations", []))
                                metadata_to_show.update({
                                    "UMLS Terms": umls_count,
                                    "Medical Relations": relations_count
                                })
                        
                        if model_type == "Conversational RAG":
                            metadata_to_show.update({
                                "Session ID": st.session_state.session_id[:8] + "...",
                                "Conversation Enabled": result.get("conversation_enabled", False),
                            })
                        
                        st.json(metadata_to_show)
                    
                    # Display evaluation results if available
                    if result.get("llm_evaluation_result"):
                        with st.expander("🔍 Evaluation Result"):
                            st.write(result["llm_evaluation_result"])
                    
                    # Display errors or warnings
                    if result.get("conversation_error"):
                        st.warning(f"⚠️ Conversation Error: {result['conversation_error']}")
                        if result.get("fallback_used"):
                            st.info("ℹ️ Fallback to standard RAG was used.")
                    
                    if result.get("medical_error"):
                        st.warning(f"⚠️ Medical Processing Error: {result['medical_error']}")
                        st.info("ℹ️ Fallback to standard RAG was used.")
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    logger.error(f"Error in main app: {e}")
                    
                    # Try fallback to standard RAG
                    try:
                        from inference_pipeline import LLM_RAG
                        fallback_model = LLM_RAG()
                        fallback_result = fallback_model.generate(
                            query=query,
                            enable_rag=enable_rag,
                            enable_evaluation=False,
                            enable_monitoring=False
                        )
                        st.info("🔄 Fallback to Standard RAG was successful")
                        st.markdown("### 🤖 Assistant Response (Fallback)")
                        st.markdown(fallback_result["answer"])
                    except Exception as fallback_error:
                        st.error(f"❌ Fallback also failed: {str(fallback_error)}")
        else:
            st.warning("⚠️ Please enter a question.")

    # Additional features in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("🔧 Advanced Features")
        
        if model_type == "Conversational RAG" and st.session_state.use_conversation:
            try:
                conv_model = ConversationalLLM_RAG()
                
                if st.button("📊 Memory Stats"):
                    stats = conv_model.get_memory_stats()
                    st.json(stats)
                
                if st.button("👤 User Profile"):
                    profile = conv_model.get_user_profile(st.session_state.user_id)
                    st.json(profile)
                
                if st.button("🗑️ Clear Session"):
                    conv_model.clear_session(st.session_state.session_id)
                    st.session_state.conversation_history = []
                    st.success("Session cleared!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error with advanced features: {e}")

if __name__ == "__main__":
    main()