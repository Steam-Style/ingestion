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

    def _normalize_features(self, output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            features = output
        elif hasattr(output, "pooler_output") and getattr(output, "pooler_output") is not None:
            features = getattr(output, "pooler_output")
        elif hasattr(output, "last_hidden_state") and getattr(output, "last_hidden_state") is not None:
            features = getattr(output, "last_hidden_state")[:, 0]
        else:
            raise TypeError(
                f"Unsupported model output type: {type(output).__name__}")

        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def get_text_embedding(self, text: str) -> Optional[Embedding]:
        if not self.is_ready():
            return None

        processor = self.processor
        model = self.model
        assert processor is not None
        assert model is not None

        try:
            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                output = model.get_text_features(**inputs)

            features = self._normalize_features(output)
            return features.squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            logger.error("Error getting text embedding: %s", exc)
            return None

    def get_image_embedding(self, image: Image.Image) -> Optional[Embedding]:
        if not self.is_ready():
            return None

        processor = self.processor
        model = self.model
        assert processor is not None
        assert model is not None

        try:
            if image.mode != "RGB":
                if image.mode == "P" and isinstance(image.info.get("transparency"), bytes):
                    image = image.convert("RGBA")
                image = image.convert("RGB")

            inputs = processor(
                images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model.get_image_features(**inputs)

            features = self._normalize_features(output)
            return features.squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            logger.error("Error getting image embedding: %s", exc)
            return None

    def get_image_embeddings(self, images: List[Image.Image]) -> List[Optional[Embedding]]:
        if not self.is_ready():
            return [None for _ in images]

        if not images:
            return []

        processor = self.processor
        model = self.model
        assert processor is not None
        assert model is not None

        try:
            prepared_images: List[Image.Image] = []

            for image in images:
                current = image
                if current.mode != "RGB":
                    if current.mode == "P" and isinstance(current.info.get("transparency"), bytes):
                        current = current.convert("RGBA")
                    current = current.convert("RGB")
                prepared_images.append(current)

            inputs = processor(images=prepared_images,
                               return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model.get_image_features(**inputs)

            features = self._normalize_features(output)
            return features.detach().cpu().tolist()
        except Exception as exc:
            logger.error("Error getting image embeddings batch: %s", exc)
            return [None for _ in images]
