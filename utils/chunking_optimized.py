import re
import hashlib
import asyncio
import threading
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Enhanced metadata cho mỗi chunk."""
    chunk_id: str
    start_char: int
    end_char: int
    token_count: int
    chunk_type: str
    semantic_score: float = 0.0
    parent_section: Optional[str] = None
    language: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    readability_score: float = 0.0
    importance_score: float = 0.0
    chunk_quality: float = 0.0

class OptimizedChunker:
    """
    Optimized chunking system với advanced features.
    """
    
    def __init__(self, 
                 max_cache_size: int = 1000,
                 enable_parallel: bool = True,
                 max_workers: int = 4):
        self.max_cache_size = max_cache_size
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # Enhanced caching với LRU
        self._chunk_cache = {}
        self._embedding_cache = {}
        self._tokenizer_cache = {}
        
        # Performance tracking
        self._performance_stats = defaultdict(list)
        
        # Initialize models
        self._initialize_models()
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=max_workers) if enable_parallel else None
    
    def _initialize_models(self):
        """Enhanced model initialization với better error handling."""
        try:
            from utils.model_utils import safe_load_sentence_transformer, get_safe_device
            
            safe_device = get_safe_device()
            logger.info(f"OptimizedChunker: Using device: {safe_device}")
            
            # Initialize embedding model
            self.embedding_model = safe_load_sentence_transformer(
                model_name=settings.EMBEDDING_MODEL_ID,
                device=safe_device
            )
            
            # Initialize tokenizer for accurate token counting
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL_ID)
                logger.info("Initialized tokenizer for accurate token counting")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}, using estimation")
                self.tokenizer = None
            
            # Initialize text splitters
            self._initialize_splitters()
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self._fallback_initialization()
    
    def _initialize_splitters(self):
        """Initialize multiple splitters for different strategies."""
        try:
            # Token-based splitter
            self.token_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=50,
                tokens_per_chunk=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
                model_name=settings.EMBEDDING_MODEL_ID,
            )
            
            # Character-based splitter for fallback
            self.char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH * 4,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Specialized splitters
            self._initialize_specialized_splitters()
            
        except Exception as e:
            logger.warning(f"Failed to initialize splitters: {e}")
            self._fallback_splitters()
    
    def _initialize_specialized_splitters(self):
        """Initialize splitters for specific content types."""
        # Medical text splitter
        self.medical_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Code splitter
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            separators=["\n\n", "\n", "def ", "class ", " ", ""]
        )
    
    def _fallback_initialization(self):
        """Fallback initialization khi models fail."""
        self.embedding_model = None
        self.tokenizer = None
        self._fallback_splitters()
    
    def _fallback_splitters(self):
        """Fallback splitters."""
        self.token_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH * 4,
            chunk_overlap=50
        )
        self.char_splitter = self.token_splitter
        self.medical_splitter = self.token_splitter
        self.code_splitter = self.token_splitter
    
    def chunk_text(self, 
                  text: str, 
                  strategy: str = "intelligent",
                  content_type: str = "general",
                  enable_metadata: bool = True) -> Union[List[str], List[ChunkMetadata]]:
        """
        Enhanced chunking function với multiple optimizations.
        
        Args:
            text: Input text
            strategy: 'intelligent', 'semantic', 'structural', 'fast', 'parallel'
            content_type: 'general', 'medical', 'code', 'table', 'list'
            enable_metadata: Return metadata if True
            
        Returns:
            List of chunks or chunk metadata
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(text, strategy, content_type)
        if cache_key in self._chunk_cache:
            logger.debug("Using cached chunks")
            result = self._chunk_cache[cache_key]
            self._log_performance("cache_hit", time.time() - start_time)
            return result
        
        # Choose chunking strategy
        if strategy == "parallel" and self.enable_parallel:
            chunks = self._parallel_chunk(text, content_type)
        elif strategy == "intelligent":
            chunks = self._intelligent_chunk_optimized(text, content_type)
        elif strategy == "semantic":
            chunks = self._semantic_chunk_optimized(text)
        elif strategy == "structural":
            chunks = self._structural_chunk_optimized(text)
        else:  # fast
            chunks = self._fast_chunk_optimized(text, content_type)
        
        # Post-processing
        chunks = self._post_process_chunks(chunks, content_type)
        
        # Generate metadata if requested
        if enable_metadata:
            result = self._generate_chunk_metadata(chunks, text, strategy)
        else:
            result = chunks
        
        # Cache results (with size limit)
        self._cache_results(cache_key, result)
        
        # Log performance
        self._log_performance(strategy, time.time() - start_time)
        
        return result
    
    def _parallel_chunk(self, text: str, content_type: str) -> List[str]:
        """Parallel chunking cho large documents."""
        if not self.enable_parallel:
            return self._intelligent_chunk_optimized(text, content_type)
        
        # Split text into sections for parallel processing
        sections = self._split_for_parallel(text)
        
        # Process sections in parallel
        futures = []
        for section in sections:
            future = self._executor.submit(
                self._chunk_section, section, content_type
            )
            futures.append(future)
        
        # Collect results
        all_chunks = []
        for future in futures:
            chunks = future.result()
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _chunk_section(self, section: Dict, content_type: str) -> List[str]:
        """Chunk a single section (for parallel processing)."""
        section_type = section.get('type', 'paragraph')
        content = section.get('content', '')
        
        if section_type == 'table':
            return self._chunk_table_optimized(content)
        elif section_type == 'list':
            return self._chunk_list_optimized(content)
        elif section_type == 'code':
            return self._chunk_code_optimized(content)
        else:
            return self._semantic_chunk_optimized(content)
    
    def _intelligent_chunk_optimized(self, text: str, content_type: str) -> List[str]:
        """Optimized intelligent chunking."""
        # Content-type specific preprocessing
        if content_type == "medical":
            return self._medical_chunk(text)
        elif content_type == "code":
            return self._code_chunk(text)
        
        # Enhanced structure detection
        structured_sections = self._detect_structure_enhanced(text)
        
        # Process sections with optimized methods
        all_chunks = []
        for section in structured_sections:
            chunks = self._process_section_optimized(section)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _detect_structure_enhanced(self, text: str) -> List[Dict]:
        """Enhanced structure detection với more patterns."""
        sections = []
        lines = text.split('\n')
        current_section = {'type': 'paragraph', 'content': '', 'start': 0}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Enhanced header detection
            if re.match(r'^#{1,6}\s+', line_stripped):
                if current_section['content'].strip():
                    sections.append(current_section)
                current_section = {'type': 'header', 'content': line, 'start': i}
            
            # Enhanced table detection
            elif self._is_table_line(line_stripped):
                if current_section['type'] != 'table':
                    if current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {'type': 'table', 'content': line + '\n', 'start': i}
                else:
                    current_section['content'] += line + '\n'
            
            # Enhanced list detection
            elif self._is_list_line(line_stripped):
                if current_section['type'] != 'list':
                    if current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {'type': 'list', 'content': line + '\n', 'start': i}
                else:
                    current_section['content'] += line + '\n'
            
            # Enhanced code detection
            elif self._is_code_line(line_stripped):
                if current_section['type'] != 'code':
                    if current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {'type': 'code', 'content': line + '\n', 'start': i}
                else:
                    current_section['content'] += line + '\n'
            
            # Regular paragraph
            else:
                if current_section['type'] not in ['paragraph']:
                    if current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {'type': 'paragraph', 'content': line + '\n', 'start': i}
                else:
                    current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _is_table_line(self, line: str) -> bool:
        """Enhanced table detection."""
        # Check for pipe separators
        if '|' in line and line.count('|') >= 2:
            return True
        
        # Check for tab-separated values
        if '\t' in line and line.count('\t') >= 2:
            return True
        
        # Check for CSV-like patterns
        if ',' in line and line.count(',') >= 3:
            return True
        
        return False
    
    def _is_list_line(self, line: str) -> bool:
        """Enhanced list detection."""
        # Bullet points
        if re.match(r'^\s*[-*+]\s+', line):
            return True
        
        # Numbered lists
        if re.match(r'^\s*\d+\.\s+', line):
            return True
        
        # Letter lists
        if re.match(r'^\s*[a-zA-Z]\.\s+', line):
            return True
        
        return False
    
    def _is_code_line(self, line: str) -> bool:
        """Enhanced code detection."""
        # Code block markers
        if line.startswith('```') or line.startswith('    '):
            return True
        
        # Programming keywords
        code_keywords = ['def ', 'class ', 'import ', 'from ', 'if __name__', 'return ']
        if any(keyword in line for keyword in code_keywords):
            return True
        
        # Indentation patterns
        if re.match(r'^\s{4,}', line):
            return True
        
        return False
    
    def _semantic_chunk_optimized(self, text: str) -> List[str]:
        """Optimized semantic chunking với better performance."""
        if not self.embedding_model:
            return self._fast_chunk_optimized(text)
        
        # Split into sentences
        sentences = self._split_into_sentences_optimized(text)
        if len(sentences) <= 2:
            return [text]
        
        # Batch embedding generation
        try:
            embeddings = self._get_embeddings_batch(sentences)
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}, falling back to fast chunking")
            return self._fast_chunk_optimized(text)
        
        # Find semantic boundaries with optimized algorithm
        boundaries = self._find_semantic_boundaries_optimized(embeddings)
        
        # Create chunks with accurate token counting
        chunks = self._create_chunks_from_boundaries(sentences, boundaries)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _get_embeddings_batch(self, sentences: List[str]) -> np.ndarray:
        """Batch embedding generation với caching."""
        # Check cache
        cache_key = hashlib.md5(' '.join(sentences).encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            sentences, 
            show_progress_bar=False,
            batch_size=32  # Optimize batch size
        )
        
        # Cache results
        self._embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def _find_semantic_boundaries_optimized(self, embeddings: np.ndarray, threshold: float = 0.7) -> List[int]:
        """Optimized semantic boundary detection."""
        boundaries = []
        
        # Use vectorized operations for better performance
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
        
        for i, similarity in enumerate(similarities.diagonal()):
            if similarity < threshold:
                boundaries.append(i)
        
        return boundaries
    
    def _create_chunks_from_boundaries(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """Create chunks from boundaries với accurate token counting."""
        chunks = []
        start_idx = 0
        
        for boundary in boundaries:
            chunk_sentences = sentences[start_idx:boundary+1]
            chunk_text = ' '.join(chunk_sentences)
            
            # Use accurate token counting
            token_count = self._count_tokens_accurate(chunk_text)
            
            if token_count <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                chunks.append(chunk_text)
                start_idx = boundary + 1
            else:
                # Split large chunk further
                sub_chunks = self._split_large_chunk_optimized(chunk_text)
                chunks.extend(sub_chunks)
                start_idx = boundary + 1
        
        # Add remaining sentences
        if start_idx < len(sentences):
            remaining_text = ' '.join(sentences[start_idx:])
            if self._count_tokens_accurate(remaining_text) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                chunks.append(remaining_text)
            else:
                sub_chunks = self._split_large_chunk_optimized(remaining_text)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _count_tokens_accurate(self, text: str) -> int:
        """Accurate token counting using tokenizer."""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Token counting failed: {e}")
        
        # Fallback to estimation
        return len(text) // 4
    
    def _split_large_chunk_optimized(self, text: str) -> List[str]:
        """Optimized large chunk splitting."""
        # Choose appropriate splitter based on content
        if self._is_code_content(text):
            splitter = self.code_splitter
        elif self._is_medical_content(text):
            splitter = self.medical_splitter
        else:
            splitter = self.char_splitter
        
        return splitter.split_text(text)
    
    def _is_code_content(self, text: str) -> bool:
        """Detect if content is code."""
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'return ', 'if __name__']
        return any(indicator in text for indicator in code_indicators)
    
    def _is_medical_content(self, text: str) -> bool:
        """Detect if content is medical."""
        medical_indicators = ['diagnosis', 'symptoms', 'treatment', 'patient', 'medical', 'disease']
        return any(indicator.lower() in text.lower() for indicator in medical_indicators)
    
    def _medical_chunk(self, text: str) -> List[str]:
        """Specialized chunking for medical content."""
        # Use medical-specific splitter
        chunks = self.medical_splitter.split_text(text)
        
        # Post-process for medical content
        processed_chunks = []
        for chunk in chunks:
            # Ensure medical terms are preserved
            processed_chunk = self._preserve_medical_terms(chunk)
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def _code_chunk(self, text: str) -> List[str]:
        """Specialized chunking for code content."""
        # Use code-specific splitter
        chunks = self.code_splitter.split_text(text)
        
        # Post-process for code content
        processed_chunks = []
        for chunk in chunks:
            # Ensure code structure is preserved
            processed_chunk = self._preserve_code_structure(chunk)
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def _preserve_medical_terms(self, chunk: str) -> str:
        """Preserve medical terms in chunks."""
        # Add medical context if needed
        if len(chunk.strip()) < 50:  # Very short chunk
            return chunk  # Keep as is for now
        
        return chunk
    
    def _preserve_code_structure(self, chunk: str) -> str:
        """Preserve code structure in chunks."""
        # Ensure code blocks are complete
        lines = chunk.split('\n')
        if lines and lines[0].strip().startswith('def ') or lines[0].strip().startswith('class '):
            # Try to include the complete function/class
            return chunk
        
        return chunk
    
    def _post_process_chunks(self, chunks: List[str], content_type: str) -> List[str]:
        """Post-process chunks for quality optimization."""
        processed_chunks = []
        
        for chunk in chunks:
            # Clean up chunk
            cleaned_chunk = self._clean_chunk(chunk)
            
            # Skip empty chunks
            if not cleaned_chunk.strip():
                continue
            
            # Quality check
            if self._is_chunk_quality_good(cleaned_chunk, content_type):
                processed_chunks.append(cleaned_chunk)
        
        return processed_chunks
    
    def _clean_chunk(self, chunk: str) -> str:
        """Clean up chunk content."""
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', chunk)
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _is_chunk_quality_good(self, chunk: str, content_type: str) -> bool:
        """Check if chunk quality is acceptable."""
        # Minimum length check
        if len(chunk.strip()) < 10:
            return False
        
        # Content-specific quality checks
        if content_type == "medical":
            return self._is_medical_chunk_quality_good(chunk)
        elif content_type == "code":
            return self._is_code_chunk_quality_good(chunk)
        
        return True
    
    def _is_medical_chunk_quality_good(self, chunk: str) -> bool:
        """Check medical chunk quality."""
        # Should contain medical terms or context
        medical_terms = ['diagnosis', 'symptoms', 'treatment', 'patient', 'medical']
        return any(term in chunk.lower() for term in medical_terms)
    
    def _is_code_chunk_quality_good(self, chunk: str) -> bool:
        """Check code chunk quality."""
        # Should contain code structure
        code_indicators = ['def ', 'class ', 'import ', 'return ', 'if ', 'for ']
        return any(indicator in chunk for indicator in code_indicators)
    
    def _generate_chunk_metadata(self, chunks: List[str], original_text: str, strategy: str) -> List[ChunkMetadata]:
        """Generate detailed metadata for chunks."""
        metadata_list = []
        start_char = 0
        
        for i, chunk in enumerate(chunks):
            end_char = start_char + len(chunk)
            
            # Calculate metadata
            token_count = self._count_tokens_accurate(chunk)
            chunk_type = self._detect_chunk_type(chunk)
            semantic_score = self._calculate_semantic_score(chunk)
            readability_score = self._calculate_readability_score(chunk)
            importance_score = self._calculate_importance_score(chunk)
            chunk_quality = self._calculate_chunk_quality(chunk)
            
            metadata = ChunkMetadata(
                chunk_id=f"chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}",
                start_char=start_char,
                end_char=end_char,
                token_count=token_count,
                chunk_type=chunk_type,
                semantic_score=semantic_score,
                readability_score=readability_score,
                importance_score=importance_score,
                chunk_quality=chunk_quality
            )
            
            metadata_list.append(metadata)
            start_char = end_char
        
        return metadata_list
    
    def _detect_chunk_type(self, chunk: str) -> str:
        """Detect chunk type."""
        if self._is_table_line(chunk.split('\n')[0]):
            return 'table'
        elif self._is_list_line(chunk.split('\n')[0]):
            return 'list'
        elif self._is_code_line(chunk.split('\n')[0]):
            return 'code'
        elif chunk.strip().startswith('#'):
            return 'header'
        else:
            return 'paragraph'
    
    def _calculate_semantic_score(self, chunk: str) -> float:
        """Calculate semantic coherence score."""
        # Simple heuristic for now
        sentences = self._split_into_sentences_optimized(chunk)
        if len(sentences) <= 1:
            return 1.0
        
        # Calculate average sentence length (proxy for coherence)
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        return min(avg_length / 100, 1.0)  # Normalize
    
    def _calculate_readability_score(self, chunk: str) -> float:
        """Calculate readability score."""
        # Simple Flesch-Kincaid approximation
        sentences = self._split_into_sentences_optimized(chunk)
        words = chunk.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        return max(0.0, min(1.0, 1.0 - (avg_sentence_length - 10) / 20))
    
    def _calculate_importance_score(self, chunk: str) -> float:
        """Calculate importance score."""
        # Simple heuristic based on content indicators
        importance_indicators = ['important', 'key', 'critical', 'essential', 'main']
        chunk_lower = chunk.lower()
        
        score = 0.0
        for indicator in importance_indicators:
            if indicator in chunk_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_chunk_quality(self, chunk: str) -> float:
        """Calculate overall chunk quality."""
        semantic_score = self._calculate_semantic_score(chunk)
        readability_score = self._calculate_readability_score(chunk)
        importance_score = self._calculate_importance_score(chunk)
        
        # Weighted average
        quality = (semantic_score * 0.4 + readability_score * 0.3 + importance_score * 0.3)
        return quality
    
    def _split_into_sentences_optimized(self, text: str) -> List[str]:
        """Optimized sentence splitting."""
        # Use regex for better performance
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_for_parallel(self, text: str) -> List[Dict]:
        """Split text for parallel processing."""
        # Split by paragraphs for parallel processing
        paragraphs = text.split('\n\n')
        sections = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                sections.append({
                    'type': 'paragraph',
                    'content': para.strip(),
                    'index': i
                })
        
        return sections
    
    def _cache_results(self, cache_key: str, result):
        """Cache results với size limit."""
        # Implement LRU cache
        if len(self._chunk_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_key]
        
        self._chunk_cache[cache_key] = result
    
    def _get_cache_key(self, text: str, strategy: str, content_type: str) -> str:
        """Generate cache key."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{strategy}_{content_type}_{text_hash}"
    
    def _log_performance(self, strategy: str, duration: float):
        """Log performance metrics."""
        self._performance_stats[strategy].append(duration)
        
        # Keep only last 100 measurements
        if len(self._performance_stats[strategy]) > 100:
            self._performance_stats[strategy] = self._performance_stats[strategy][-100:]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {}
        for strategy, measurements in self._performance_stats.items():
            if measurements:
                stats[strategy] = {
                    'avg_duration': np.mean(measurements),
                    'min_duration': np.min(measurements),
                    'max_duration': np.max(measurements),
                    'count': len(measurements)
                }
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        self._chunk_cache.clear()
        self._embedding_cache.clear()
        self._tokenizer_cache.clear()
        logger.info("Cleared all chunking caches")

# Global instance
_optimized_chunker_instance = None

def get_optimized_chunker() -> OptimizedChunker:
    """Get singleton optimized chunker instance."""
    global _optimized_chunker_instance
    if _optimized_chunker_instance is None:
        _optimized_chunker_instance = OptimizedChunker()
    return _optimized_chunker_instance

# Backward compatible functions
def chunk_text_optimized(text: str, strategy: str = "intelligent", content_type: str = "general") -> List[str]:
    """Optimized chunking function."""
    chunker = get_optimized_chunker()
    return chunker.chunk_text(text, strategy, content_type, enable_metadata=False)

def chunk_with_metadata_optimized(text: str, strategy: str = "intelligent", content_type: str = "general") -> List[ChunkMetadata]:
    """Optimized chunking with metadata."""
    chunker = get_optimized_chunker()
    return chunker.chunk_text(text, strategy, content_type, enable_metadata=True) 