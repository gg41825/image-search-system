from typing import List
import torch
from transformers import BertTokenizer, BertModel
import config

class BERTEmbedder:
    """
    Wraps BERT (bert-base-uncased) to produce text embeddings.
    Returns L2-normalized tensors of shape [N, hidden_size].
    """

    def __init__(self, model_name: str = config.MODEL_BERT, device: str | None = None):
        # Select device: prefer MPS (Apple Silicon GPU), fallback to CPU
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # important: disable dropout etc. during inference

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into embeddings.
        Strategy: use [CLS] token embedding (first token).
        Returns:
            torch.Tensor: [N, hidden_size] tensor (L2 normalized).
        """
        # Tokenize batch and move to device
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)

        # Use [CLS] embedding as sentence representation
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # Normalize embeddings for cosine similarity
        return torch.nn.functional.normalize(cls_emb, p=2, dim=1)
