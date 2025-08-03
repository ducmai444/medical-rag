from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import torch
import numpy as np
from typing import List, Optional, Union
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UMLS_CrossEncoder:
    """
    Cross-encoder model for scoring UMLS concept pairs.
    
    This class implements a cross-encoder model specifically designed for scoring
    the relevance between medical queries and UMLS concepts.
    
    Attributes:
        model_name (str): Name or path of the cross-encoder model
        max_length (int): Maximum sequence length for tokenization
        batch_size (int): Batch size for inference
        device (str): Device to run the model on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Cross-Encoder",
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
        max_chars: int = 500
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_chars = max_chars
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        try:
            logger.info(f"Loading model {model_name} on {self.device}")
            self.model = CrossEncoder(model_name, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def prepare_pairs(self, query: str, rels: List[str]) -> List[List[str]]:
        """
        Prepare and validate query-relation pairs for scoring.
        
        Args:
            query (str): The query text
            rels (List[str]): List of relation texts
            
        Returns:
            List[List[str]]: List of [query, relation] pairs
        """
        pairs = []
        for rel in rels:
            # Skip empty relations
            if not rel.strip():
                continue
                
            # Truncate relation if needed
            truncated_rel = rel[:self.max_chars]
            if len(rel) > self.max_chars:
                logger.debug(f"Truncated relation from {len(rel)} to {self.max_chars} chars")
                
            # Create and validate pair
            pair = [query, truncated_rel]
            tokens = self.tokenizer.encode(query, truncated_rel, add_special_tokens=True)
            if len(tokens) > self.max_length:
                logger.warning(
                    f"Token length {len(tokens)} exceeds max_length {self.max_length}\n"
                    f"Query: {query[:50]}...\nRel: {truncated_rel[:50]}..."
                )
            
            pairs.append(pair)
            
        return pairs

    def score_batch(self, pairs: List[List[str]]) -> np.ndarray:
        """
        Score a batch of query-relation pairs.
        
        Args:
            pairs (List[List[str]]): List of [query, relation] pairs
            
        Returns:
            np.ndarray: Array of scores
        """
        try:
            # Tokenize
            encodings = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get scores
            with torch.no_grad():
                logits = self.model.model(**encodings).logits
                scores = logits.cpu().numpy().flatten()
                
            return scores
            
        except Exception as e:
            logger.error(f"Error in batch scoring: {str(e)}")
            return np.array([0.0] * len(pairs))  # Return zero scores on error

    def score(self, query: str, rels: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Score the relevance between a query and multiple relations.
        
        Args:
            query (str): The query text
            rels (List[str]): List of relation texts to score against the query
            show_progress (bool): Whether to show a progress bar for large batches
            
        Returns:
            np.ndarray: Array of relevance scores
        """
        # Input validation
        if not rels:
            logger.warning("Empty relations list provided")
            return np.array([])
            
        if not query.strip():
            logger.warning("Empty query provided")
            return np.array([0.0] * len(rels))
        
        try:
            # Prepare pairs
            pairs = self.prepare_pairs(query, rels)
            if not pairs:
                return np.array([])
                
            # Score in batches
            scores = []
            batches = range(0, len(pairs), self.batch_size)
            
            # Add progress bar if requested
            if show_progress:
                batches = tqdm(batches, desc="Scoring pairs")
                
            for i in batches:
                batch_pairs = pairs[i:i + self.batch_size]
                batch_scores = self.score_batch(batch_pairs)
                scores.extend(batch_scores)
            
            return np.array(scores)
            
        except Exception as e:
            logger.error(f"Error in scoring: {str(e)}")
            return np.array([0.0] * len(rels))

    def truncate_rels(self, rels: List[str]) -> List[str]:
        """
        Truncate relations to maximum character length.
        
        Args:
            rels (List[str]): List of relations to truncate
            
        Returns:
            List[str]: List of truncated relations
        """
        return [rel[:self.max_chars] for rel in rels]

    def __call__(self, query: str, rels: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Make the class callable for convenient scoring.
        
        Args:
            query (str): The query text
            rels (List[str]): List of relations to score
            show_progress (bool): Whether to show a progress bar
            
        Returns:
            np.ndarray: Array of relevance scores
        """
        return self.score(query, rels, show_progress)

# Example usage:
# cross_encoder = UMLS_CrossEncoder()
# scores = cross_encoder.score("heart disease", ["cardiovascular disorder", "cardiac condition"])
# # Or using call syntax
# scores = cross_encoder("heart disease", ["cardiovascular disorder", "cardiac condition"])