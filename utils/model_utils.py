"""
Utility functions for safe model loading with meta tensor handling.
"""

import torch
import logging
from sentence_transformers import SentenceTransformer
from typing import Optional

logger = logging.getLogger(__name__)

def safe_load_sentence_transformer(
    model_name: str, 
    device: Optional[str] = None,
    trust_remote_code: bool = True
) -> Optional[SentenceTransformer]:
    """
    Safely load SentenceTransformer with proper meta tensor handling.
    
    Args:
        model_name: Model name or path
        device: Target device ('cuda', 'cpu', or None for auto)
        trust_remote_code: Whether to trust remote code
        
    Returns:
        SentenceTransformer instance or None if failed
    """
    
    # Determine target device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Attempting to load {model_name} on {device}")
    
    # Strategy 1: Try direct loading
    try:
        model = SentenceTransformer(
            model_name, 
            device=device,
            trust_remote_code=trust_remote_code
        )
        logger.info(f"✅ Successfully loaded {model_name} on {device}")
        return model
    except Exception as e:
        logger.warning(f"Direct loading failed: {e}")
    
    # Strategy 2: Load on CPU first, then move to target device
    try:
        logger.info("Trying CPU-first loading strategy...")
        model = SentenceTransformer(
            model_name, 
            device='cpu',
            trust_remote_code=trust_remote_code
        )
        
        if device != 'cpu':
            # Use to_empty() for meta tensor compatibility
            if hasattr(model, '_modules'):
                for module in model._modules.values():
                    if hasattr(module, 'to_empty'):
                        module.to_empty(device=device)
                    else:
                        module.to(device)
            else:
                model.to(device)
        
        logger.info(f"✅ Successfully loaded {model_name} via CPU-first strategy")
        return model
    except Exception as e:
        logger.warning(f"CPU-first strategy failed: {e}")
    
    # Strategy 3: Force CPU mode
    try:
        logger.info("Forcing CPU-only mode...")
        model = SentenceTransformer(
            model_name, 
            device='cpu',
            trust_remote_code=trust_remote_code
        )
        logger.info(f"✅ Successfully loaded {model_name} on CPU (forced)")
        return model
    except Exception as e:
        logger.error(f"All loading strategies failed: {e}")
        return None

def get_safe_device() -> str:
    """
    Get a safe device for model loading.
    
    Returns:
        Safe device string ('cuda' or 'cpu')
    """
    try:
        if torch.cuda.is_available():
            # Test CUDA availability
            torch.cuda.empty_cache()
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            return "cuda"
    except Exception as e:
        logger.warning(f"CUDA test failed: {e}")
    
    return "cpu"

def clear_model_cache():
    """Clear model cache and GPU memory."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear HuggingFace cache if needed
        import gc
        gc.collect()
        
        logger.info("Model cache cleared")
    except Exception as e:
        logger.warning(f"Cache clearing failed: {e}")

def fix_meta_tensor_model(model):
    """
    Fix model that has meta tensors by reinitializing parameters.
    """
    try:
        import torch.nn as nn
        
        def init_weights(m):
            if hasattr(m, 'weight') and m.weight is not None:
                if m.weight.is_meta:
                    # Reinitialize meta tensors
                    if isinstance(m, nn.Linear):
                        m.weight = nn.Parameter(torch.randn_like(m.weight, device='cpu'))
                        if m.bias is not None and m.bias.is_meta:
                            m.bias = nn.Parameter(torch.randn_like(m.bias, device='cpu'))
                    elif isinstance(m, nn.Embedding):
                        m.weight = nn.Parameter(torch.randn_like(m.weight, device='cpu'))
        
        model.apply(init_weights)
        logger.info("Fixed meta tensor model")
        return model
    except Exception as e:
        logger.error(f"Failed to fix meta tensor model: {e}")
        return model 