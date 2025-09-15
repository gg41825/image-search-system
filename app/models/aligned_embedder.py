# models/aligned_embedder.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoImageProcessor
import config

class AlignedEmbedder(nn.Module):
    def __init__(self,
                 text_model_name: str = config.MODEL_DINO,
                 vision_model_name: str = "facebook/dinov2-base"):
        super().__init__()
        # Text encoder (BERT)
        self.bert = BertModel.from_pretrained(text_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(text_model_name)

        # Vision encoder (DINOv2)
        self.dino = AutoModel.from_pretrained(vision_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

    def forward(self, input_ids, attention_mask, pixel_values):
        # BERT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_out.last_hidden_state[:, 0, :]  # [B, 768]

        # DINOv2
        dino_out = self.dino(pixel_values=pixel_values)
        dino_cls = getattr(dino_out, "pooler_output", None)
        if dino_cls is None:
            dino_cls = dino_out.last_hidden_state[:, 0, :]  # [B, 768]

        # Concat
        return torch.cat([bert_cls, dino_cls], dim=1)
