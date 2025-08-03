#!/usr/bin/env python3
"""
Quick test script for MedicalRAGPipeline
"""

import sys
import os
import logging

# Suppress warnings for cleaner output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

def quick_test():
    """Quick test with a single query."""
    
    print("🚀 Quick Test - MedicalRAGPipeline")
    print("=" * 50)
    
    # Import here to catch import errors
    try:
        from medical.medical_pipeline import MedicalRAGPipeline
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # UMLS API key
    UMLS_API_KEY = "b50edd64-9197-492e-8987-bc8c073c8bfa"
    
    # Test query
    test_query = "Thuốc Paracetamol có thể dùng để hạ sốt ở trẻ em bao nhiêu lần một ngày?"
    
    try:
        print("\n🔧 Initializing pipeline...")
        pipeline = MedicalRAGPipeline(
            umls_api_key=UMLS_API_KEY,
            enable_translation=True,
            enable_caching=False,
            max_workers=1
        )
        print("✅ Pipeline initialized!")
        
        print(f"\n📝 Processing query: {test_query}")
        
        # Process query
        context = pipeline.process_query(test_query)
        
        print("\n📊 Results:")
        print(f"  Original: {context.original_query}")
        print(f"  Translated: {context.translated_query}")
        print(f"  Medical terms: {context.medical_terms}")
        print(f"  UMLS results: {len(context.umls_results)}")
        print(f"  Final relations: {len(context.final_relations)}")
        print(f"  Confidence: {context.confidence_score:.2f}")
        print(f"  Processing time: {context.processing_time:.2f}s")
        
        # Show some results
        if context.umls_results:
            print("\n🏥 UMLS Terms:")
            for cui, term in list(context.umls_results.items())[:2]:  # Show first 2
                print(f"  - {term.name} (CUI: {cui})")
        
        if context.final_relations:
            print("\n🔗 Top Relations:")
            for i, (subj, rel, obj) in enumerate(context.final_relations[:3]):  # Show first 3
                print(f"  {i+1}. {subj} --[{rel}]--> {obj}")
        
        # Format for RAG
        formatted = pipeline.format_medical_context(context)
        if formatted:
            print(f"\n📄 Context Preview (first 200 chars):")
            print(f"  {formatted[:200]}...")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test() 