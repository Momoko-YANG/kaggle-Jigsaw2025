import os
import torch
from sentence_transformers import SentenceTransformer, models
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wrapper for sentence transformer model."""
    
    def __init__(self, model_path: str, max_seq_length: int = 128, 
                 use_fp16: bool = True):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.use_fp16 = use_fp16
        self.model = None
        
    def load_model(self) -> SentenceTransformer:
        """Load or initialize model."""
        try:
            word_embedding = models.Transformer(
                self.model_path,
                max_seq_length=self.max_seq_length,
                do_lower_case=True
            )
            pooling = models.Pooling(
                word_embedding.get_word_embedding_dimension(),
                pooling_mode="mean"
            )
            self.model = SentenceTransformer(modules=[word_embedding, pooling])
            
            if self.use_fp16:
                self.model.half()
            
            logger.info(f"Loaded model from {self.model_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        """Generate embeddings for texts."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        embeddings = self.model.encode(
            sentences=texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=normalize
        )
        return embeddings
    
    def save(self, output_path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
