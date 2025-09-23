from typing import List, Optional
import numpy as np
from models.base_embedder import BaseEmbedder
from models.bert_embedder import BERTEmbedder
from models.dino_embedder import DINOv2Embedder

class LocalEmbedder(BaseEmbedder):
    """
    Local embedder that combines:
    - BERT for text embeddings
    - DINOv2 for image embeddings
    Supports three modes:
        1. Text only
        2. Image only
        3. Text + Image
    If one modality is missing, a zero vector is used to keep dimensions consistent.
    """

    def __init__(self):
        # Initialize sub-models
        self.bert = BERTEmbedder()       # Text encoder (dim=768)
        self.dino = DINOv2Embedder()     # Image encoder (dim=768)
        self.text_dim = 768
        self.image_dim = 768
        self.total_dim = self.text_dim + self.image_dim

    def embed(self, texts: Optional[List[str]] = None,
                    images: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate embeddings by combining text and image encoders.

        Args:
            texts (List[str] or None): Input text strings.
            images (List[str] or None): Input image URLs or paths.

        Returns:
            np.ndarray: Concatenated embeddings [N, total_dim].
        """
        batch_size = 0
        parts = []

        # --- Encode text (if provided) ---
        if texts:
            text_emb = self.bert.embed_texts(texts).cpu().numpy()
            text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)
            batch_size = text_emb.shape[0]
        else:
            text_emb = None

        # --- Encode images (if provided) ---
        if images:
            img_emb = self.dino.embed_images(images).cpu().numpy()
            img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True)
            batch_size = img_emb.shape[0]
        else:
            img_emb = None

        # --- Handle missing modality (pad with zeros) ---
        if text_emb is None:
            text_emb = np.zeros((batch_size, self.text_dim), dtype=np.float32)
        if img_emb is None:
            img_emb = np.zeros((batch_size, self.image_dim), dtype=np.float32)

        # --- Concatenate embeddings ---
        return np.concatenate([text_emb, img_emb], axis=1)  # [N, 1536]
