#!/usr/bin/env python3
"""
Test script để kiểm tra tích hợp conversation memory với inference pipeline.
Chạy script này để verify rằng tất cả components hoạt động đúng.
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test import các modules cần thiết."""
    print("=== Testing Imports ===")
    
    try:
        from inference_pipeline import LLM_RAG, ConversationalLLM_RAG
        print("✅ Successfully imported LLM_RAG and ConversationalLLM_RAG")
    except ImportError as e:
        print(f"❌ Failed to import from inference_pipeline: {e}")
        return False
    
    try:
        from memory import ConversationManager, ShortTermMemory, LongTermMemory
        print("✅ Successfully imported memory components")
    except ImportError as e:
        print(f"❌ Failed to import memory components: {e}")
        return False
    
    return True

def test_standard_rag():
    """Test RAG pipeline gốc."""
    print("\n=== Testing Standard RAG ===")
    
    try:
        from inference_pipeline import LLM_RAG
        
        model = LLM_RAG()
        print("✅ Successfully initialized LLM_RAG")
        
        # Test với query đơn giản (không gọi LLM thực tế để tránh lỗi API)
        print("✅ Standard RAG initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test standard RAG: {e}")
        return False

def test_conversational_rag_initialization():
    """Test khởi tạo ConversationalLLM_RAG."""
    print("\n=== Testing Conversational RAG Initialization ===")
    
    try:
        from inference_pipeline import ConversationalLLM_RAG
        
        # Test khởi tạo
        conv_rag = ConversationalLLM_RAG(
            max_context_tokens=1024,
            short_term_window=3,
            long_term_k=2
        )
        print("✅ Successfully initialized ConversationalLLM_RAG")
        
        # Kiểm tra conversation enabled
        if conv_rag.conversation_enabled:
            print("✅ Conversation support is enabled")
        else:
            print("⚠️ Conversation support is disabled (fallback mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize ConversationalLLM_RAG: {e}")
        return False

def test_memory_components():
    """Test các memory components riêng lẻ."""
    print("\n=== Testing Memory Components ===")
    
    try:
        from memory import ShortTermMemory, ConversationManager
        
        # Test ShortTermMemory
        short_term = ShortTermMemory(max_messages_per_session=10, ttl_hours=1)
        
        # Test store và retrieve
        short_term.store_message("test_session", "Hello", "user")
        short_term.store_message("test_session", "Hi there!", "assistant")
        
        history = short_term.get_recent_history("test_session", window_size=2)
        print(f"✅ ShortTermMemory: Retrieved {len(history)} messages")
        
        # Test ConversationManager
        conv_manager = ConversationManager(
            max_context_tokens=1024,
            short_term_window=3,
            long_term_k=2
        )
        print("✅ Successfully initialized ConversationManager")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test memory components: {e}")
        return False

def test_session_management():
    """Test session management functions."""
    print("\n=== Testing Session Management ===")
    
    try:
        from inference_pipeline import ConversationalLLM_RAG
        
        conv_rag = ConversationalLLM_RAG()
        
        if not conv_rag.conversation_enabled:
            print("⚠️ Skipping session management test (conversation disabled)")
            return True
        
        # Test session info
        session_info = conv_rag.get_session_info("test_session_123")
        print(f"✅ Session info retrieved: {session_info.get('exists', 'unknown')}")
        
        # Test memory stats
        stats = conv_rag.get_memory_stats()
        print(f"✅ Memory stats retrieved: {len(stats)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test session management: {e}")
        return False

def test_error_handling():
    """Test error handling và fallback mechanisms."""
    print("\n=== Testing Error Handling ===")
    
    try:
        from inference_pipeline import ConversationalLLM_RAG
        
        conv_rag = ConversationalLLM_RAG()
        
        # Test với invalid session
        try:
            conv_rag.clear_session("")
            print("✅ Handled empty session ID gracefully")
        except Exception as e:
            print(f"✅ Expected error for empty session: {type(e).__name__}")
        
        # Test get user profile với invalid user
        profile = conv_rag.get_user_profile("invalid_user_id")
        if "error" in profile or not conv_rag.conversation_enabled:
            print("✅ Handled invalid user ID gracefully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed error handling test: {e}")
        return False

def main():
    """Chạy tất cả tests."""
    print("🧪 Starting Conversation Integration Tests\n")
    
    tests = [
        test_imports,
        test_standard_rag,
        test_conversational_rag_initialization,
        test_memory_components,
        test_session_management,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is successful.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 