import logging
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

Embedding = List[float]


class SiglipEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", backend: str = "torchvision") -> None:
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, backend=backend)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", model_name, exc)
            self.processor = None
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def get_text_embedding(self, text: str) -> Optional[Embedding]:
        if not self.is_ready():
            return None

        try:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                output = self.model.get_text_features(**inputs)

            features = output.pooler_output.float()
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            logger.error("Error getting text embedding: %s", exc)
            return None

    def get_image_embedding(self, image: Image.Image) -> Optional[Embedding]:
        if not self.is_ready():
            return None

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(
                images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.get_image_features(**inputs)

            features = output.pooler_output.float()
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).detach().cpu().tolist()
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Error getting image embedding: %s", exc)
            return None
