from typing import List
import numpy as np

class BaseEmbedder:
    def embed(self, texts: List[str], images: List[str]) -> np.ndarray:
        """
        Abstract method to return embeddings as numpy array [N, dim].
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Embed method must be implemented by subclasses")
