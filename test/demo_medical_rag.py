#!/usr/bin/env python3
"""
Demo script for Medical RAG Pipeline.
Tests Vietnamese medical Q&A with UMLS integration.
"""

import os
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_medical_pipeline():
    """Test the medical pipeline components."""
    print("🧪 Testing Medical RAG Pipeline\n")
    
    # Test queries (Vietnamese and English)
    test_queries = [
        # Vietnamese medical queries
        "Triệu chứng của bệnh tiểu đường type 2 là gì?",
        "Thuốc metformin có tác dụng phụ gì không?",
        "Cách điều trị cao huyết áp ở người già?",
        
        # English medical queries  
        "What are the symptoms of hypertension?",
        "How does insulin work in diabetes treatment?",
        "What are the side effects of amoxicillin?",
        
        # Mixed queries
        "Patient has fever và đau đầu, what could be the diagnosis?",
    ]
    
    # UMLS API key (you need to set this)
    umls_api_key = os.getenv("UMLS_API_KEY")
    if not umls_api_key:
        print("❌ UMLS_API_KEY environment variable not set")
        print("Please set your UMLS API key: export UMLS_API_KEY='your_key_here'")
        return False
    
    try:
        # Test individual components first
        print("1️⃣ Testing Medical Pipeline Components...")
        test_components(umls_api_key)
        
        print("\n2️⃣ Testing Full Medical RAG...")
        test_full_medical_rag(umls_api_key, test_queries)
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_components(umls_api_key: str):
    """Test individual medical components."""
    try:
        from medical import MedicalRAGPipeline
        
        # Initialize pipeline
        pipeline = MedicalRAGPipeline(umls_api_key=umls_api_key)
        
        # Test Vietnamese query
        test_query = "Triệu chứng của bệnh tiểu đường là gì?"
        
        print(f"🔍 Processing: '{test_query}'")
        
        # Process query
        medical_context = pipeline.process_query(test_query)
        
        print(f"✅ Original: {medical_context.original_query}")
        print(f"✅ Translated: {medical_context.translated_query}")
        print(f"✅ Medical Entities: {[e.term for e in medical_context.medical_entities]}")
        print(f"✅ UMLS Results: {len(medical_context.umls_results)}")
        print(f"✅ Confidence: {medical_context.confidence_score:.2f}")
        print(f"✅ Processing Time: {medical_context.processing_time:.2f}s")
        
        # Show UMLS results
        if medical_context.umls_results:
            print("\n🏥 UMLS Knowledge:")
            for i, result in enumerate(medical_context.umls_results, 1):
                print(f"  {i}. {result.name} → {result.relation_label} → {result.related_concept}")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        raise

def test_full_medical_rag(umls_api_key: str, queries: List[str]):
    """Test full Medical RAG integration."""
    try:
        from inference_pipeline import MedicalLLM_RAG
        
        # Initialize Medical RAG
        print("🏥 Initializing Medical RAG...")
        medical_rag = MedicalLLM_RAG(umls_api_key=umls_api_key)
        
        if not medical_rag.is_medical_enabled():
            print("⚠️ Medical pipeline not fully enabled")
            return
        
        print("✅ Medical RAG initialized successfully!")
        
        # Test each query
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"🔍 Test {i}: {query}")
            print('='*60)
            
            try:
                # Generate response
                result = medical_rag.generate(
                    query=query,
                    enable_rag=True,
                    enable_medical=True,
                    medical_priority_weight=0.7
                )
                
                # Display results
                print(f"🤖 **Answer:** {result['answer']}")
                print(f"⚡ **Pipeline:** {result.get('pipeline_type', 'unknown')}")
                print(f"🏥 **Medical Query:** {result.get('is_medical_query', False)}")
                
                if result.get('medical_metadata'):
                    medical_meta = result['medical_metadata']
                    print(f"🎯 **Medical Confidence:** {medical_meta['confidence_score']:.2f}")
                    
                    if medical_meta.get('medical_entities'):
                        entities = [e['term'] for e in medical_meta['medical_entities']]
                        print(f"🔬 **Entities:** {', '.join(entities)}")
                    
                    if medical_meta.get('umls_results'):
                        print(f"📚 **UMLS Results:** {len(medical_meta['umls_results'])}")
                        for umls_result in medical_meta['umls_results'][:2]:
                            print(f"   - {umls_result['name']} {umls_result['relation']} {umls_result['related_concept']}")
                
                print(f"⏱️ **Processing Time:** {result.get('processing_time', 0):.2f}s")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
                continue
        
    except Exception as e:
        print(f"❌ Full RAG test failed: {e}")
        raise

def test_comparison():
    """Compare Medical RAG vs Standard RAG."""
    print("\n🆚 Comparing Medical RAG vs Standard RAG")
    
    medical_query = "What are the side effects of metformin in diabetes treatment?"
    
    try:
        from inference_pipeline import LLM_RAG, MedicalLLM_RAG
        
        umls_api_key = os.getenv("UMLS_API_KEY")
        if not umls_api_key:
            print("❌ UMLS API key required for comparison")
            return
        
        # Standard RAG
        print("\n1️⃣ Standard RAG Response:")
        standard_rag = LLM_RAG()
        standard_result = standard_rag.generate(query=medical_query, enable_rag=True)
        print(f"📝 {standard_result['answer']}")
        
        # Medical RAG
        print("\n2️⃣ Medical RAG Response:")
        medical_rag = MedicalLLM_RAG(umls_api_key=umls_api_key)
        medical_result = medical_rag.generate(
            query=medical_query, 
            enable_rag=True, 
            enable_medical=True
        )
        print(f"🏥 {medical_result['answer']}")
        
        # Comparison
        print("\n📊 Comparison:")
        print(f"Standard Processing Time: {standard_result.get('processing_time', 0):.2f}s")
        print(f"Medical Processing Time: {medical_result.get('processing_time', 0):.2f}s")
        
        if medical_result.get('medical_metadata'):
            medical_meta = medical_result['medical_metadata']
            print(f"Medical Entities Found: {len(medical_meta.get('medical_entities', []))}")
            print(f"UMLS Knowledge Used: {len(medical_meta.get('umls_results', []))}")
            print(f"Medical Confidence: {medical_meta['confidence_score']:.2f}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

def main():
    """Main demo function."""
    print("🚀 Medical RAG Pipeline Demo")
    print("="*50)
    
    # Check environment
    if not os.getenv("UMLS_API_KEY"):
        print("⚠️ Set UMLS_API_KEY environment variable to run full demo")
        print("Example: export UMLS_API_KEY='your_umls_api_key'")
        print("\nYou can get UMLS API key from: https://uts.nlm.nih.gov/uts/")
        return 1
    
    # Run tests
    success = test_medical_pipeline()
    
    if success:
        # Run comparison
        test_comparison()
        print("\n🎉 Demo completed successfully!")
        return 0
    else:
        print("\n❌ Demo failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 