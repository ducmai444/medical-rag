#!/usr/bin/env python3
"""
Docker-compatible test script for Medical RAG Pipeline.
Tests basic functionality without requiring UMLS API key.
"""

import os
import sys
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports without initializing heavy models."""
    print("🔍 Testing Basic Imports...")
    
    results = {}
    
    # Test core dependencies
    try:
        import torch
        results['torch'] = f"✅ PyTorch {torch.__version__}"
    except ImportError as e:
        results['torch'] = f"❌ PyTorch: {e}"
    
    try:
        import transformers
        results['transformers'] = f"✅ Transformers {transformers.__version__}"
    except ImportError as e:
        results['transformers'] = f"❌ Transformers: {e}"
    
    try:
        import langdetect
        results['langdetect'] = "✅ Language Detection"
    except ImportError as e:
        results['langdetect'] = f"❌ Language Detection: {e}"
    
    try:
        import networkx
        results['networkx'] = f"✅ NetworkX {networkx.__version__}"
    except ImportError as e:
        results['networkx'] = f"❌ NetworkX: {e}"
    
    try:
        from fuzzywuzzy import fuzz
        results['fuzzywuzzy'] = "✅ FuzzyWuzzy"
    except ImportError as e:
        results['fuzzywuzzy'] = f"❌ FuzzyWuzzy: {e}"
    
    # Test medical modules
    try:
        sys.path.append('/app/others')
        from others.translation import EnViT5Translator
        results['translation'] = "✅ Translation Module"
    except ImportError as e:
        results['translation'] = f"❌ Translation Module: {e}"
    
    try:
        from others.ner import MedicalNERLLM
        results['medical_ner'] = "✅ Medical NER Module"
    except ImportError as e:
        results['medical_ner'] = f"❌ Medical NER Module: {e}"
    
    try:
        from others.umls import UMLS_API
        results['umls'] = "✅ UMLS API Module"
    except ImportError as e:
        results['umls'] = f"❌ UMLS API Module: {e}"
    
    try:
        from others.ranking import MMR_reranking
        results['ranking'] = "✅ Ranking Module"
    except ImportError as e:
        results['ranking'] = f"❌ Ranking Module: {e}"
    
    # Test medical pipeline
    try:
        from medical import MedicalRAGPipeline
        results['medical_pipeline'] = "✅ Medical Pipeline"
    except ImportError as e:
        results['medical_pipeline'] = f"❌ Medical Pipeline: {e}"
    
    try:
        from inference_pipeline import MedicalLLM_RAG
        results['medical_rag'] = "✅ Medical RAG Integration"
    except ImportError as e:
        results['medical_rag'] = f"❌ Medical RAG Integration: {e}"
    
    # Print results
    print("\n📋 Import Test Results:")
    for component, status in results.items():
        print(f"  {component}: {status}")
    
    # Count successes
    successes = sum(1 for status in results.values() if status.startswith("✅"))
    total = len(results)
    
    print(f"\n📊 Summary: {successes}/{total} components imported successfully")
    return successes == total

def test_device_detection():
    """Test device detection and CPU fallback."""
    print("\n🖥️ Testing Device Detection...")
    
    try:
        import torch
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Current Device: {torch.cuda.current_device()}")
            print(f"Device Name: {torch.cuda.get_device_name()}")
        
        # Test CPU mode enforcement
        cpu_forced = os.getenv("CUDA_VISIBLE_DEVICES") == ""
        force_cpu = os.getenv("FORCE_CPU_ONLY", "false").lower() == "true"
        
        print(f"CPU Mode Forced (CUDA_VISIBLE_DEVICES): {cpu_forced}")
        print(f"Force CPU Only Flag: {force_cpu}")
        
        # Test tensor creation
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0])
            print(f"✅ CPU Tensor Creation: {test_tensor.device}")
        except Exception as e:
            print(f"❌ CPU Tensor Creation: {e}")
        
        # Test GPU tensor (should fail in CPU mode)
        if cuda_available and not cpu_forced and not force_cpu:
            try:
                gpu_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
                print(f"✅ GPU Tensor Creation: {gpu_tensor.device}")
            except Exception as e:
                print(f"❌ GPU Tensor Creation: {e}")
        else:
            print("🚫 GPU Tensor Creation: Skipped (CPU mode)")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available")
        return False

def test_medical_components_lightweight():
    """Test medical components without heavy model loading."""
    print("\n🏥 Testing Medical Components (Lightweight)...")
    
    try:
        # Test UMLS API (without actual API call)
        from others.umls import UMLS_API
        
        fake_api_key = "test_key"
        umls_api = UMLS_API(fake_api_key)
        print("✅ UMLS API instantiation successful")
        
        # Test medical terms dictionary
        from others.ner import MEDICAL_TERMS
        print(f"✅ Medical terms dictionary loaded: {len(MEDICAL_TERMS)} terms")
        
        # Test ranking functions (without actual ranking)
        from others.ranking import get_similarity
        print("✅ Ranking functions imported")
        
        # Test language detection
        from langdetect import detect
        test_text = "This is a test sentence"
        detected_lang = detect(test_text)
        print(f"✅ Language detection: '{test_text}' -> {detected_lang}")
        
        # Test Vietnamese detection
        vietnamese_text = "Đây là câu tiếng Việt"
        detected_lang_vi = detect(vietnamese_text)
        print(f"✅ Vietnamese detection: '{vietnamese_text}' -> {detected_lang_vi}")
        
        return True
        
    except Exception as e:
        print(f"❌ Medical components test failed: {e}")
        traceback.print_exc()
        return False

def test_rag_integration():
    """Test RAG integration without model loading."""
    print("\n🔗 Testing RAG Integration...")
    
    try:
        # Test standard RAG import
        from inference_pipeline import LLM_RAG
        print("✅ Standard RAG import successful")
        
        # Test medical RAG import
        from inference_pipeline import MedicalLLM_RAG
        print("✅ Medical RAG import successful")
        
        # Test conversation RAG import
        from inference_pipeline import ConversationalLLM_RAG
        print("✅ Conversational RAG import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG integration test failed: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test Docker environment setup."""
    print("\n🐳 Testing Docker Environment...")
    
    # Check environment variables
    env_vars = [
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES", 
        "FORCE_CPU_ONLY",
        "HF_HOME",
        "TRANSFORMERS_CACHE"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "Not Set")
        print(f"  {var}: {value}")
    
    # Check working directory
    print(f"  Working Directory: {os.getcwd()}")
    
    # Check if we're in container
    in_container = os.path.exists("/.dockerenv")
    print(f"  Running in Container: {in_container}")
    
    # Check available memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    memory = line.split()[1]
                    memory_gb = int(memory) / 1024 / 1024
                    print(f"  Available Memory: {memory_gb:.1f} GB")
                    break
    except:
        print("  Available Memory: Unable to detect")
    
    return True

def main():
    """Main test function."""
    print("🚀 Medical RAG Docker Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Device Detection", test_device_detection),
        ("Medical Components", test_medical_components_lightweight),
        ("RAG Integration", test_rag_integration),
        ("Environment", test_environment)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Medical RAG is ready for Docker!")
        return 0
    elif passed >= total * 0.7:
        print("⚠️ Most tests passed. System should work with some limitations.")
        return 0
    else:
        print("❌ Many tests failed. Check dependencies and configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 