from typing import List
import numpy as np
from models.base_embedder import BaseEmbedder
from models.bert_embedder import BERTEmbedder
from models.dino_embedder import DINOv2Embedder
    
class LocalEmbedder(BaseEmbedder):
    def __init__(self):
        self.bert = BERTEmbedder()
        self.dino = DINOv2Embedder()

    def embed(self, texts: List[str], images: List[str]) -> np.ndarray:
        text_emb = self.bert.embed_texts(texts).cpu().numpy()
        img_emb = self.dino.embed_images(images).cpu().numpy()

        # normalize
        text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)
        img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True)

        # concat
        return np.concatenate([text_emb, img_emb], axis=1)
