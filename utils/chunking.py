import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import logging

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata cho mỗi chunk."""
    chunk_id: str
    start_char: int
    end_char: int
    token_count: int
    chunk_type: str  # 'paragraph', 'header', 'list', 'table', 'code'
    semantic_score: float = 0.0
    parent_section: Optional[str] = None

class IntelligentChunker:
    """
    Intelligent chunking system với semantic awareness và structure detection.
    """
    
    def __init__(self):
        self.embedding_model = None
        self.token_splitter = None
        self._initialize_models()
        
        # Caching để tối ưu performance
        self._chunk_cache = {}
        self._embedding_cache = {}
        
    def _initialize_models(self):
        """Khởi tạo models một lần duy nhất."""
        try:
            from utils.model_utils import safe_load_sentence_transformer, get_safe_device
            
            # Get safe device
            safe_device = get_safe_device()
            logger.info(f"Chunking: Using safe device: {safe_device}")
            
            # Khởi tạo embedding model với safe loading
            self.embedding_model = safe_load_sentence_transformer(
                model_name=settings.EMBEDDING_MODEL_ID,
                device=safe_device
            )
            
            if self.embedding_model is not None:
                logger.info(f"Initialized chunking embedding model on {safe_device}")
            else:
                logger.warning("Failed to load embedding model for chunking, semantic features disabled")
            
            # Khởi tạo token splitter (reuse)
            try:
                self.token_splitter = SentenceTransformersTokenTextSplitter(
                    chunk_overlap=50,
                    tokens_per_chunk=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
                    model_name=settings.EMBEDDING_MODEL_ID,
                )
                logger.info("Initialized token splitter")
            except Exception as e:
                logger.warning(f"Failed to initialize token splitter: {e}, using basic splitter")
                self.token_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH * 4,
                    chunk_overlap=50
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize chunking models: {e}")
            # Fallback: basic text splitter
            self.embedding_model = None
            self.token_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH * 4,  # Estimate
                chunk_overlap=50
            )
    
    def chunk_text(self, text: str, strategy: str = "intelligent") -> List[str]:
        """
        Main chunking function với multiple strategies.
        
        Args:
            text: Input text
            strategy: 'intelligent', 'semantic', 'structural', 'fast'
            
        Returns:
            List of text chunks
        """
        # Check cache
        cache_key = self._get_cache_key(text, strategy)
        if cache_key in self._chunk_cache:
            logger.debug("Using cached chunks")
            return self._chunk_cache[cache_key]
        
        # Choose chunking strategy
        if strategy == "intelligent":
            chunks = self._intelligent_chunk(text)
        elif strategy == "semantic":
            chunks = self._semantic_chunk(text)
        elif strategy == "structural":
            chunks = self._structural_chunk(text)
        else:  # fast
            chunks = self._fast_chunk(text)
        
        # Cache results
        self._chunk_cache[cache_key] = chunks
        return chunks
    
    def _intelligent_chunk(self, text: str) -> List[str]:
        """
        Intelligent chunking: Structure + Semantic + Optimization.
        """
        # Step 1: Detect document structure
        structured_sections = self._detect_structure(text)
        
        # Step 2: Apply appropriate chunking to each section
        all_chunks = []
        for section in structured_sections:
            if section['type'] in ['header', 'title']:
                # Headers: keep with next section
                all_chunks.append(section['content'])
            elif section['type'] == 'table':
                # Tables: preserve structure, split by rows if too large
                table_chunks = self._chunk_table(section['content'])
                all_chunks.extend(table_chunks)
            elif section['type'] == 'list':
                # Lists: group related items
                list_chunks = self._chunk_list(section['content'])
                all_chunks.extend(list_chunks)
            elif section['type'] == 'code':
                # Code: preserve logical blocks
                code_chunks = self._chunk_code(section['content'])
                all_chunks.extend(code_chunks)
            else:
                # Regular paragraphs: semantic chunking
                para_chunks = self._semantic_chunk_paragraph(section['content'])
                all_chunks.extend(para_chunks)
        
        # Step 3: Post-process và optimize
        optimized_chunks = self._optimize_chunks(all_chunks)
        
        return optimized_chunks
    
    def _detect_structure(self, text: str) -> List[Dict]:
        """
        Detect document structure (headers, tables, lists, code blocks).
        """
        sections = []
        lines = text.split('\n')
        current_section = {'type': 'paragraph', 'content': '', 'start': 0}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Detect headers (markdown style)
            if re.match(r'^#{1,6}\s+', line_stripped):
                if current_section['content'].strip():
                    sections.append(current_section)
                current_section = {'type': 'header', 'content': line, 'start': i}
            
            # Detect tables (simple heuristic)
            elif '|' in line and line.count('|') >= 2:
                if current_section['type'] != 'table':
                    if current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {'type': 'table', 'content': line + '\n', 'start': i}
                else:
                    current_section['content'] += line + '\n'
            
            # Detect lists
            elif re.match(r'^\s*[-*+]\s+', line_stripped) or re.match(r'^\s*\d+\.\s+', line_stripped):
                if current_section['type'] != 'list':
                    if current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {'type': 'list', 'content': line + '\n', 'start': i}
                else:
                    current_section['content'] += line + '\n'
            
            # Detect code blocks
            elif line_stripped.startswith('```') or line_stripped.startswith('    '):
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
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Semantic-aware chunking using sentence embeddings.
        """
        if not self.embedding_model:
            return self._fast_chunk(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 2:
            return [text]
        
        # Get embeddings for sentences
        try:
            embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}, falling back to fast chunking")
            return self._fast_chunk(text)
        
        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(embeddings)
        
        # Create chunks based on boundaries
        chunks = []
        start_idx = 0
        
        for boundary in boundaries:
            chunk_sentences = sentences[start_idx:boundary+1]
            chunk_text = ' '.join(chunk_sentences)
            
            # Check token limit
            if self._estimate_tokens(chunk_text) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                chunks.append(chunk_text)
                start_idx = boundary + 1
            else:
                # Split large chunk further
                sub_chunks = self._split_large_chunk(chunk_text)
                chunks.extend(sub_chunks)
                start_idx = boundary + 1
        
        # Add remaining sentences
        if start_idx < len(sentences):
            remaining_text = ' '.join(sentences[start_idx:])
            if self._estimate_tokens(remaining_text) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                chunks.append(remaining_text)
            else:
                sub_chunks = self._split_large_chunk(remaining_text)
                chunks.extend(sub_chunks)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _structural_chunk(self, text: str) -> List[str]:
        """
        Structure-aware chunking (preserve paragraphs, sections).
        """
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph exceeds token limit
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if self._estimate_tokens(test_chunk) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single paragraph is too large, split it
                if self._estimate_tokens(para) > settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                    sub_chunks = self._split_large_chunk(para)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _fast_chunk(self, text: str) -> List[str]:
        """
        Fast chunking using token splitter (optimized version of original).
        """
        try:
            return self.token_splitter.split_text(text)
        except Exception as e:
            logger.warning(f"Token splitting failed: {e}, using character splitting")
            # Fallback to character splitting
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH * 4,  # Rough estimate
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return char_splitter.split_text(text)
    
    # Helper methods
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_semantic_boundaries(self, embeddings: np.ndarray, threshold: float = 0.7) -> List[int]:
        """Find semantic boundaries using cosine similarity."""
        boundaries = []
        
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            if similarity < threshold:  # Low similarity = potential boundary
                boundaries.append(i)
        
        return boundaries
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """Split a large chunk that exceeds token limit."""
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH * 3,  # Conservative estimate
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return char_splitter.split_text(text)
    
    def _chunk_table(self, table_text: str) -> List[str]:
        """Intelligent table chunking."""
        lines = table_text.strip().split('\n')
        if len(lines) <= 5:  # Small table
            return [table_text]
        
        # Split large tables by rows, keeping header
        chunks = []
        header = lines[0] if lines else ""
        
        current_chunk = header
        for line in lines[1:]:
            test_chunk = current_chunk + '\n' + line
            if self._estimate_tokens(test_chunk) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                current_chunk = test_chunk
            else:
                chunks.append(current_chunk)
                current_chunk = header + '\n' + line
        
        if current_chunk != header:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_list(self, list_text: str) -> List[str]:
        """Intelligent list chunking."""
        items = re.split(r'\n(?=\s*[-*+]\s+|\s*\d+\.\s+)', list_text)
        
        chunks = []
        current_chunk = ""
        
        for item in items:
            test_chunk = current_chunk + '\n' + item if current_chunk else item
            if self._estimate_tokens(test_chunk) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = item
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_code(self, code_text: str) -> List[str]:
        """Intelligent code chunking."""
        # Simple code chunking by functions/classes
        lines = code_text.split('\n')
        chunks = []
        current_chunk = ""
        
        for line in lines:
            test_chunk = current_chunk + '\n' + line if current_chunk else line
            if self._estimate_tokens(test_chunk) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _semantic_chunk_paragraph(self, paragraph: str) -> List[str]:
        """Apply semantic chunking to a single paragraph."""
        if self._estimate_tokens(paragraph) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
            return [paragraph]
        
        return self._semantic_chunk(paragraph)
    
    def _optimize_chunks(self, chunks: List[str]) -> List[str]:
        """Post-process optimization: merge small chunks, split large ones."""
        optimized = []
        current_chunk = ""
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Try to merge with current chunk
            test_merge = current_chunk + '\n\n' + chunk if current_chunk else chunk
            
            if self._estimate_tokens(test_merge) <= settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                current_chunk = test_merge
            else:
                # Save current chunk
                if current_chunk:
                    optimized.append(current_chunk)
                
                # Check if new chunk needs splitting
                if self._estimate_tokens(chunk) > settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH:
                    sub_chunks = self._split_large_chunk(chunk)
                    optimized.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = chunk
        
        # Add final chunk
        if current_chunk:
            optimized.append(current_chunk)
        
        return optimized
    
    def _get_cache_key(self, text: str, strategy: str) -> str:
        """Generate cache key for text and strategy."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{strategy}_{text_hash}"


# Global instance để reuse models
_chunker_instance = None

def get_chunker() -> IntelligentChunker:
    """Get singleton chunker instance."""
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = IntelligentChunker()
    return _chunker_instance

# Backward compatible function
def chunk_text(text: str, strategy: str = "intelligent") -> List[str]:
    """
    Main chunking function (backward compatible).
    
    Args:
        text: Input text
        strategy: 'intelligent', 'semantic', 'structural', 'fast'
    
    Returns:
        List of text chunks
    """
    chunker = get_chunker()
    return chunker.chunk_text(text, strategy)

# Additional utility functions
def chunk_with_metadata(text: str, strategy: str = "intelligent") -> List[ChunkMetadata]:
    """
    Chunk text and return with metadata.
    """
    chunker = get_chunker()
    chunks = chunker.chunk_text(text, strategy)
    
    metadata_list = []
    start_char = 0
    
    for i, chunk in enumerate(chunks):
        end_char = start_char + len(chunk)
        
        metadata = ChunkMetadata(
            chunk_id=f"chunk_{i}",
            start_char=start_char,
            end_char=end_char,
            token_count=chunker._estimate_tokens(chunk),
            chunk_type="auto_detected"
        )
        
        metadata_list.append(metadata)
        start_char = end_char
    
    return metadata_list