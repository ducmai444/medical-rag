#!/usr/bin/env python3
"""
Test script for MedicalRAGPipeline
"""

import sys
import os
import logging
import time
from pprint import pprint

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'medical'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'others'))

from medical.medical_pipeline import MedicalRAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_medical_pipeline():
    """Test MedicalRAGPipeline with various queries."""
    
    # UMLS API key from notebook
    UMLS_API_KEY = "b50edd64-9197-492e-8987-bc8c073c8bfa"
    
    print("🔬 Initializing MedicalRAGPipeline...")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = MedicalRAGPipeline(
            umls_api_key=UMLS_API_KEY,
            enable_translation=True,
            enable_caching=False,  # Disable Redis for testing
            max_workers=2  # Reduce workers for testing
        )
        
        print("✅ Pipeline initialized successfully!")
        print(f"📊 Pipeline stats: {pipeline.get_stats()}")
        print("\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Test cases from notebook
    test_queries = [
        {
            "name": "Vietnamese Medical Query 1",
            "query": "Thuốc Paracetamol có thể dùng để hạ sốt ở trẻ em bao nhiêu lần một ngày?",
            "expected_terms": ["paracetamol", "fever", "children"]
        },
        {
            "name": "Vietnamese Medical Query 2", 
            "query": "Bệnh cao huyết áp có thể dẫn đến những biến chứng nguy hiểm nào nếu không điều trị?",
            "expected_terms": ["hypertension", "complications"]
        },
        {
            "name": "English Medical Query",
            "query": "How does obesity contribute to type 2 diabetes in individuals with a sedentary lifestyle?",
            "expected_terms": ["obesity", "type 2 diabetes", "diabetes"]
        },
        {
            "name": "Mixed Medical Query",
            "query": "Patient has sốt cao và đau đầu",
            "expected_terms": ["fever", "headache", "pain"]
        }
    ]
    
    # Test each query
    for i, test_case in enumerate(test_queries, 1):
        print(f"🧪 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        print(f"📝 Query: {test_case['query']}")
        
        try:
            # Check if medical query
            is_medical = pipeline.is_medical_query(test_case['query'])
            print(f"🔍 Is medical query: {is_medical}")
            
            # Process query
            start_time = time.time()
            context = pipeline.process_query(test_case['query'])
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print(f"🎯 Confidence score: {context.confidence_score:.2f}")
            
            # Display results
            print("\n📋 Results:")
            print(f"  Original query: {context.original_query}")
            print(f"  Translated query: {context.translated_query}")
            print(f"  Medical terms: {context.medical_terms}")
            print(f"  UMLS results count: {len(context.umls_results)}")
            print(f"  Final relations count: {len(context.final_relations)}")
            
            # Show UMLS results
            if context.umls_results:
                print("\n🏥 UMLS Results:")
                for cui, term in context.umls_results.items():
                    print(f"  - {term.name} (CUI: {cui})")
                    if term.definition:
                        print(f"    Definition: {term.definition[:100]}...")
                    print(f"    Relations: {len(term.relations)}")
            
            # Show final relations
            if context.final_relations:
                print("\n🔗 Final Relations (Top 5):")
                for i, (subj, rel, obj) in enumerate(context.final_relations[:5]):
                    print(f"  {i+1}. ({subj}) --[{rel}]--> ({obj})")
            
            # Format for RAG
            formatted_context = pipeline.format_medical_context(context)
            if formatted_context:
                print(f"\n📄 Formatted Context Preview:")
                preview = formatted_context[:300] + "..." if len(formatted_context) > 300 else formatted_context
                print(f"  {preview}")
            
            # Get metadata
            metadata = pipeline.get_medical_metadata(context)
            print(f"\n📊 Metadata:")
            print(f"  Medical terms count: {len(metadata['medical_terms'])}")
            print(f"  UMLS terms count: {len(metadata['umls_terms'])}")
            print(f"  Final relations count: {len(metadata['final_relations'])}")
            print(f"  Translation used: {metadata['translation_used']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60 + "\n")

def test_individual_components():
    """Test individual components separately."""
    print("🔧 Testing Individual Components...")
    print("=" * 60)
    
    UMLS_API_KEY = "b50edd64-9197-492e-8987-bc8c073c8bfa"
    
    try:
        pipeline = MedicalRAGPipeline(
            umls_api_key=UMLS_API_KEY,
            enable_translation=True,
            enable_caching=False,
            max_workers=1
        )
        
        test_query = "Thuốc paracetamol để hạ sốt"
        
        # Test translation
        print("1. Testing Translation:")
        translated = pipeline._translate_query(test_query)
        print(f"   Original: {test_query}")
        print(f"   Translated: {translated}")
        
        # Test medical NER
        print("\n2. Testing Medical NER:")
        terms = pipeline._extract_medical_terms(translated)
        print(f"   Medical terms: {terms}")
        
        # Test single term processing
        if terms:
            print(f"\n3. Testing Single Term Processing: '{terms[0]}'")
            result = pipeline._process_single_term(terms[0], translated)
            if result:
                cui, medical_term = result
                print(f"   CUI: {cui}")
                print(f"   Name: {medical_term.name}")
                print(f"   Definition: {medical_term.definition[:100]}...")
                print(f"   Relations count: {len(medical_term.relations)}")
                print(f"   Ranked relations count: {len(medical_term.ranked_relations)}")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()

def test_error_handling():
    """Test error handling scenarios."""
    print("⚠️  Testing Error Handling...")
    print("=" * 60)
    
    try:
        # Test with invalid API key
        print("1. Testing with invalid API key:")
        pipeline = MedicalRAGPipeline(
            umls_api_key="invalid-key",
            enable_translation=False,
            enable_caching=False,
            max_workers=1
        )
        
        context = pipeline.process_query("test query")
        print(f"   Result: confidence={context.confidence_score}, terms={len(context.medical_terms)}")
        
    except Exception as e:
        print(f"   Expected error: {e}")
    
    print("\n2. Testing with empty query:")
    try:
        UMLS_API_KEY = "b50edd64-9197-492e-8987-bc8c073c8bfa"
        pipeline = MedicalRAGPipeline(
            umls_api_key=UMLS_API_KEY,
            enable_translation=False,
            enable_caching=False,
            max_workers=1
        )
        
        context = pipeline.process_query("")
        print(f"   Result: confidence={context.confidence_score}, terms={len(context.medical_terms)}")
        
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting MedicalRAGPipeline Tests")
    print("=" * 60)
    
    # Main test
    test_medical_pipeline()
    
    # Component tests
    test_individual_components()
    
    # Error handling tests
    test_error_handling()
    
    print("✅ All tests completed!") 