from typing import List, Union, Optional
import io
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import time
import config

class DINOv2Embedder:
    """
    Minimal wrapper around Facebook's DINOv2 ViT models loaded via torch.hub.
    Produces L2-normalized global embeddings from the CLS token.

    Model sizes and output dims (approx):
      - dinov2_vits14  -> 384
      - dinov2_vitb14  -> 768   (good default)
      - dinov2_vitl14  -> 1024
      - dinov2_vitg14  -> 1536
    """

    def __init__(
        self,
        model_name: str = config.MODEL_DINO,
        device: Optional[str] = None,
        image_size: int = 518,
        timeout: int = 10,
    ):
        # Prefer MPS on Apple Silicon, else CPU
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        # Load model from torch.hub (downloads on first run)
        # Repo: facebookresearch/dinov2
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, trust_repo=True
        ).to(self.device)
        self.model.eval()

        # Preprocessing (DINOv2 uses ImageNet mean/std, 518x518 center crop by default)
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.timeout = timeout

    @torch.no_grad()
    def embed_images(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """
        Encode a batch of images into global DINOv2 embeddings (L2-normalized).
        Args:
            images: list of URLs, file paths, or PIL.Image
        Returns:
            Tensor of shape [N, D]
        """
        pil_batch = [self._to_pil(img) for img in images]
        pixel_batch = torch.stack([self.preprocess(im) for im in pil_batch]).to(self.device)
        # print(pixel_batch)

        # Prefer using forward_features if available to get CLS token explicitly
        feats = self._forward_to_embedding(pixel_batch)
        feats = F.normalize(feats, p=2, dim=1)
        return feats

    # -------------------------------
    # Internals
    # -------------------------------
    def _to_pil(self, x: Union[str, Image.Image]) -> Image.Image:
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, str):
            if x.startswith("http://") or x.startswith("https://"):
                resp = requests.get(x, timeout=self.timeout)
                time.sleep(1)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            return Image.open(x).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(x)}")

    @torch.no_grad()
    def _forward_to_embedding(self, pixel_batch: torch.Tensor) -> torch.Tensor:
        """
        Get the CLS/global embedding. DINOv2 torch.hub models expose `forward_features`
        that returns a dict with keys like 'x_norm_clstoken'. If not present,
        fallback to plain forward() output.
        """
        # Some hub models: dict with 'x_norm_clstoken'
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(pixel_batch)
            if isinstance(out, dict):
                # Preferred key in official repo
                if "x_norm_clstoken" in out:
                    return out["x_norm_clstoken"]
                # Some variants might just return 'x_norm'
                if "x_norm" in out and out["x_norm"].ndim == 2:
                    return out["x_norm"]
        # Fallback: assume model(pixel_batch) already gives [N, D]
        feats = self.model(pixel_batch)
        if feats.ndim == 3:
            # If [N, tokens, D], take CLS token at index 0
            feats = feats[:, 0, :]
        return feats
