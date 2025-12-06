import os
import numpy as np
import requests
import torch
from transformers import BertTokenizer, BertModel, AutoImageProcessor, AutoModel
from PIL import Image
import config

class TritonEmbedder:
    """
    Embedder using PyTorch for inference (no ONNX Runtime).
    CPU-optimized for Cloud Run deployment.
    """

    def __init__(self, url: str = None, model_name: str = None):
        """
        Initialize the embedder with PyTorch models.
        
        Args:
            url: Kept for API compatibility, not used
            model_name: Kept for API compatibility, not used
        """
        print("Initializing TritonEmbedder with PyTorch (CPU mode)")
        
        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_BERT)
        self.bert_model = BertModel.from_pretrained(config.MODEL_BERT)
        self.bert_model.eval()
        
        # Load DINOv2 model and processor
        self.image_processor = AutoImageProcessor.from_pretrained(config.TRIOTON_MODEL_DINO)
        self.dino_model = AutoModel.from_pretrained(config.TRIOTON_MODEL_DINO)
        self.dino_model.eval()
        
        print("TritonEmbedder initialized with PyTorch")

    def _load_image(self, path_or_url: str) -> Image.Image:
        """
        Support both local file path and HTTP URL.
        
        Args:
            path_or_url: Either a local file path or HTTP/HTTPS URL
            
        Returns:
            PIL Image object in RGB format
            
        Raises:
            ValueError: If the path/URL is invalid
        """
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")
        elif os.path.exists(path_or_url):
            return Image.open(path_or_url).convert("RGB")
        else:
            raise ValueError(f"Invalid image path or URL: {path_or_url}")

    def embed(self, text: str, image_path_or_url: str) -> np.ndarray:
        """
        Generate aligned embedding using PyTorch models.

        Args:
            text: The input text string for BERT
            image_path_or_url: Either a local path or HTTP URL for DINOv2

        Returns:
            A 2D numpy array of shape [1, embedding_dim]
        """
        with torch.no_grad():
            # Preprocess text
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=16
            )
            
            # Get BERT embedding
            bert_output = self.bert_model(**tokens)
            text_embedding = bert_output.last_hidden_state[:, 0, :].numpy()
            
            # Preprocess image
            image = self._load_image(image_path_or_url)
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
            
            # Get DINOv2 embedding
            dino_output = self.dino_model(pixel_values)
            image_embedding = dino_output.last_hidden_state[:, 0, :].numpy()
            
            # Concatenate embeddings
            combined = np.concatenate([text_embedding, image_embedding], axis=1)
            
            return combined