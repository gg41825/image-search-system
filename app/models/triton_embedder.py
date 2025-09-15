import numpy as np
import requests
from typing import List

class TritonEmbedder:
    def __init__(self, url: str = "http://localhost:8000", model_name: str = "aligned"):
        self.url = url.rstrip("/")
        self.model_name = model_name

        # Triton inference endpoint
        self.infer_url = f"{self.url}/v2/models/{self.model_name}/infer"

    def embed(self, text: str, image_url: str) -> np.ndarray:
        """
        Call Triton Inference Server to get aligned embedding.
        Here we assume Triton model takes already preprocessed inputs (dummy placeholders for now).
        """
        payload = {
            "inputs": [
                {"name": "input_ids", "shape": [1, 16], "datatype": "INT64", "data": [1] * 16},
                {"name": "attention_mask", "shape": [1, 16], "datatype": "INT64", "data": [1] * 16},
                {"name": "pixel_values", "shape": [1, 3, 224, 224], "datatype": "FP32",
                 "data": [0.0] * (1 * 3 * 224 * 224)}
            ],
            "outputs": [{"name": "embedding"}]
        }

        resp = requests.post(self.infer_url, json=payload)
        resp.raise_for_status()
        result = resp.json()["outputs"][0]["data"]

        return np.array(result).reshape(1, -1)
