#!/usr/bin/env python3
"""
Test script để demo Intelligent Chunking System.
"""

import time
from utils.chunking import chunk_text, chunk_with_metadata, get_chunker

# Sample texts for testing different strategies
SAMPLE_TEXTS = {
    "simple": """
This is a simple paragraph. It contains multiple sentences. 
Each sentence should be processed correctly.

This is another paragraph. It has different content.
The chunking system should handle this appropriately.
""",
    
    "structured": """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
- Classification
- Regression
- Decision Trees

### Unsupervised Learning  
- Clustering
- Association Rules
- Dimensionality Reduction

## Code Example

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Data Table

| Algorithm | Type | Use Case |
|-----------|------|----------|
| Linear Regression | Supervised | Prediction |
| K-Means | Unsupervised | Clustering |
| Decision Tree | Supervised | Classification |
""",
    
    "long_paragraph": """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems (such as chess and Go), autonomously operating vehicles, intelligent routing in content delivery networks, and military simulations.
"""
}

def test_chunking_strategies():
    """Test different chunking strategies."""
    print("🧪 Testing Intelligent Chunking System\n")
    
    for text_name, text in SAMPLE_TEXTS.items():
        print(f"📄 Testing with: {text_name.upper()}")
        print("=" * 50)
        
        strategies = ["fast", "structural", "semantic", "intelligent"]
        
        for strategy in strategies:
            start_time = time.time()
            
            try:
                chunks = chunk_text(text, strategy=strategy)
                end_time = time.time()
                
                print(f"\n🔧 Strategy: {strategy}")
                print(f"⏱️  Time: {end_time - start_time:.3f}s")
                print(f"📊 Chunks: {len(chunks)}")
                
                for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                    print(f"   Chunk {i+1}: {chunk[:100]}...")
                
                if len(chunks) > 2:
                    print(f"   ... and {len(chunks) - 2} more chunks")
                    
            except Exception as e:
                print(f"❌ Error with {strategy}: {e}")
        
        print("\n" + "="*70 + "\n")

def test_chunking_with_metadata():
    """Test chunking với metadata."""
    print("📋 Testing Chunking with Metadata\n")
    
    text = SAMPLE_TEXTS["structured"]
    
    try:
        metadata_list = chunk_with_metadata(text, strategy="intelligent")
        
        print(f"📊 Total chunks: {len(metadata_list)}")
        print("\n📋 Chunk Metadata:")
        
        for i, metadata in enumerate(metadata_list[:3]):  # Show first 3
            print(f"\nChunk {i+1}:")
            print(f"  ID: {metadata.chunk_id}")
            print(f"  Type: {metadata.chunk_type}")
            print(f"  Tokens: {metadata.token_count}")
            print(f"  Chars: {metadata.start_char}-{metadata.end_char}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_performance():
    """Test performance với different text sizes."""
    print("⚡ Performance Testing\n")
    
    # Create texts of different sizes
    base_text = SAMPLE_TEXTS["long_paragraph"]
    sizes = [1, 5, 10, 20]  # Multipliers
    
    for size in sizes:
        test_text = base_text * size
        text_length = len(test_text)
        
        print(f"📏 Text size: {text_length:,} characters")
        
        for strategy in ["fast", "intelligent"]:
            start_time = time.time()
            
            try:
                chunks = chunk_text(test_text, strategy=strategy)
                end_time = time.time()
                
                print(f"  {strategy:12}: {end_time - start_time:.3f}s ({len(chunks)} chunks)")
                
            except Exception as e:
                print(f"  {strategy:12}: Error - {e}")
        
        print()

def test_caching():
    """Test caching performance."""
    print("💾 Caching Test\n")
    
    text = SAMPLE_TEXTS["structured"]
    
    # First run (no cache)
    start_time = time.time()
    chunks1 = chunk_text(text, strategy="intelligent")
    first_run_time = time.time() - start_time
    
    # Second run (with cache)
    start_time = time.time()
    chunks2 = chunk_text(text, strategy="intelligent")
    second_run_time = time.time() - start_time
    
    print(f"🔄 First run:  {first_run_time:.3f}s ({len(chunks1)} chunks)")
    print(f"💾 Second run: {second_run_time:.3f}s ({len(chunks2)} chunks)")
    print(f"🚀 Speedup:    {first_run_time / second_run_time:.1f}x")
    print(f"✅ Same result: {chunks1 == chunks2}")

def test_model_initialization():
    """Test model initialization và error handling."""
    print("🔧 Model Initialization Test\n")
    
    try:
        chunker = get_chunker()
        
        print(f"✅ Chunker initialized")
        print(f"🤖 Embedding model available: {chunker.embedding_model is not None}")
        print(f"🔤 Token splitter available: {chunker.token_splitter is not None}")
        
        # Test with simple text
        test_text = "This is a simple test. It should work fine."
        chunks = chunker.chunk_text(test_text, strategy="intelligent")
        print(f"📊 Test chunking: {len(chunks)} chunks")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")

def main():
    """Run all tests."""
    print("🚀 Intelligent Chunking System - Comprehensive Test\n")
    
    tests = [
        test_model_initialization,
        test_chunking_strategies,
        test_chunking_with_metadata,
        test_caching,
        test_performance
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
        
        print("\n" + "="*80 + "\n")
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main() 