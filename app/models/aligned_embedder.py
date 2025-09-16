# models/aligned_embedder.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoImageProcessor
import config

class AlignedEmbedder(nn.Module):
    """
    AlignedEmbedder combines a text encoder (BERT) and a vision encoder (DINOv2)
    into a single PyTorch module.

    Purpose:
    - This module is used when exporting to ONNX/Triton.
    - It outputs a single aligned embedding vector that concatenates text + image features.

    Input:
        input_ids: token IDs from BERT tokenizer [B, seq_len]
        attention_mask: attention mask for BERT [B, seq_len]
        pixel_values: preprocessed image tensor [B, 3, H, W]

    Output:
        embedding: concatenated representation [B, 768 + 768]
    """

    def __init__(self,
                 text_model_name: str = config.MODEL_BERT,
                 vision_model_name: str = "facebook/dinov2-base"):
        super().__init__()

        # --- Text encoder (BERT) ---
        self.bert = BertModel.from_pretrained(text_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(text_model_name)

        # --- Vision encoder (DINOv2) ---
        self.dino = AutoModel.from_pretrained(vision_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

    def forward(self, input_ids, attention_mask, pixel_values):
        """
        Forward pass through both encoders and align outputs.

        Args:
            input_ids (torch.Tensor): Token IDs for text input [B, seq_len].
            attention_mask (torch.Tensor): Attention mask [B, seq_len].
            pixel_values (torch.Tensor): Preprocessed image tensor [B, 3, H, W].

        Returns:
            torch.Tensor: Concatenated embedding [B, 1536].
        """

        # --- Text embedding (CLS token from BERT) ---
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_out.last_hidden_state[:, 0, :]  # [B, 768]

        # --- Image embedding (CLS token from DINOv2) ---
        dino_out = self.dino(pixel_values=pixel_values)
        dino_cls = getattr(dino_out, "pooler_output", None)
        if dino_cls is None:
            dino_cls = dino_out.last_hidden_state[:, 0, :]  # fallback [B, 768]

        # --- Concatenate embeddings ---
        return torch.cat([bert_cls, dino_cls], dim=1)  # [B, 1536]