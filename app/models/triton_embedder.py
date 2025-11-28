import os
import numpy as np
import requests
from transformers import BertTokenizer, AutoImageProcessor
from PIL import Image
import config

class TritonEmbedder:
    """
    A client-side wrapper to send text + image inputs to Triton Inference Server
    and receive aligned embeddings from the deployed ONNX model.
    """

    def __init__(self, url: str = config.TRITON_URL, model_name: str = config.TRITON_MODEL_NAME):
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.infer_url = f"{self.url}/v2/models/{self.model_name}/infer"

        # Local preprocessing tools
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_BERT)
        self.image_processor = AutoImageProcessor.from_pretrained(config.TRIOTON_MODEL_DINO)

    def _load_image(self, path_or_url: str) -> Image.Image:
        """Support both local file path and HTTP URL"""
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")
        elif os.path.exists(path_or_url):
            return Image.open(path_or_url).convert("RGB")
        else:
            raise ValueError(f"❌ Invalid image path or URL: {path_or_url}")

    def embed(self, text: str, image_path_or_url: str) -> np.ndarray:
        """
        Generate aligned embedding by sending inputs to Triton.

        Args:
            text (str): The input text string for BERT.
            image_path_or_url (str): Either a local path or HTTP URL for DINOv2.

        Returns:
            np.ndarray: A 2D numpy array of shape [1, embedding_dim].
        """

        # --- Preprocess text ---
        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=16
        )
        input_ids = tokens["input_ids"].astype("int64")
        attention_mask = tokens["attention_mask"].astype("int64")

        # --- Preprocess image ---
        image = self._load_image(image_path_or_url)
        pixel_values = self.image_processor(images=image, return_tensors="np")["pixel_values"].astype("float32")

        # --- Triton payload ---
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(input_ids.shape),
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist()
                },
                {
                    "name": "attention_mask",
                    "shape": list(attention_mask.shape),
                    "datatype": "INT64",
                    "data": attention_mask.flatten().tolist()
                },
                {
                    "name": "pixel_values",
                    "shape": list(pixel_values.shape),
                    "datatype": "FP32",
                    "data": pixel_values.flatten().tolist()
                }
            ],
            "outputs": [{"name": "embedding"}]
        }

        resp = requests.post(self.infer_url, json=payload)
        resp.raise_for_status()
        result = resp.json()["outputs"][0]["data"]

        return np.array(result).reshape(1, -1)