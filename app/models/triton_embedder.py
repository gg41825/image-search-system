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
        # Hugging Face tokenizer for text (BERT-based)
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_BERT)
        # Hugging Face image processor (e.g., DINOv2 preprocessor)
        self.image_processor = AutoImageProcessor.from_pretrained(config.TRIOTON_MODEL_DINO)

    def embed(self, text: str, image_url: str) -> np.ndarray:
        """
        Generate aligned embedding by sending inputs to Triton.

        Args:
            text (str): The input text string for BERT.
            image_url (str): The image URL for DINOv2.

        Returns:
            np.ndarray: A 2D numpy array of shape [1, embedding_dim].
        """

        # --- Preprocess text ---
        # Tokenize text into input_ids and attention_mask (numpy format, int64)
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
        # Download and convert image to RGB
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        # Use Hugging Face processor to normalize and resize image to tensor
        pixel_values = self.image_processor(images=image, return_tensors="np")["pixel_values"].astype("float32")

        # --- Triton payload ---
        # Build inference request payload following Triton HTTP/JSON protocol
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

        # --- Send request ---
        # Call Triton inference API
        resp = requests.post(self.infer_url, json=payload)
        resp.raise_for_status()
        result = resp.json()["outputs"][0]["data"]

        # Convert flat list back into numpy array [1, dim]
        return np.array(result).reshape(1, -1)
