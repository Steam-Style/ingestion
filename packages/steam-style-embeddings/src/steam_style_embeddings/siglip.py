import logging

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    SiglipTextModel,
    SiglipVisionModel,
)

logger = logging.getLogger(__name__)

Embedding = list[float]


class SiglipEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        backend: str = "torchvision",
        load_text: bool = True,
        load_vision: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.load_text = load_text
        self.load_vision = load_vision
        self.processor = None
        self.model = None
        self.max_text_length: int | None = None

        try:
            # Only load the parts that will be used, the text tower and tokenizer are most of the memory
            if load_text and load_vision:
                self.processor = AutoProcessor.from_pretrained(
                    model_name, backend=backend)
                self.model = AutoModel.from_pretrained(model_name)
            elif load_text:
                self.processor = AutoTokenizer.from_pretrained(model_name)
                self.model = SiglipTextModel.from_pretrained(model_name)
            elif load_vision:
                self.processor = AutoImageProcessor.from_pretrained(
                    model_name, backend=backend)
                self.model = SiglipVisionModel.from_pretrained(model_name)
            else:
                raise ValueError(
                    "At least one of load_text and load_vision must be set")

            # The tokenizer alone does not pad to the length the model was trained with
            if load_text:
                text_config = getattr(
                    self.model.config, "text_config", self.model.config)
                self.max_text_length = text_config.max_position_embeddings

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
        elif hasattr(output, "pooler_output") and output.pooler_output is not None:
            features = output.pooler_output
        elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            features = output.last_hidden_state[:, 0]
        else:
            raise TypeError(
                f"Unsupported model output type: {type(output).__name__}")

        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _get_text_features(self, inputs: dict) -> object:
        if isinstance(self.model, SiglipTextModel):
            return self.model(**inputs)
        return self.model.get_text_features(**inputs)

    def _get_image_features(self, inputs: dict) -> object:
        if isinstance(self.model, SiglipVisionModel):
            return self.model(**inputs)
        return self.model.get_image_features(**inputs)

    def get_text_embedding(self, text: str) -> Embedding | None:
        if not self.is_ready() or not self.load_text:
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
                max_length=self.max_text_length,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                output = self._get_text_features(inputs)

            features = self._normalize_features(output)
            return features.squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            logger.error("Error getting text embedding: %s", exc)
            return None

    def get_image_embedding(self, image: Image.Image) -> Embedding | None:
        if not self.is_ready() or not self.load_vision:
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
                output = self._get_image_features(inputs)

            features = self._normalize_features(output)
            return features.squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            logger.error("Error getting image embedding: %s", exc)
            return None

    def get_image_embeddings(self, images: list[Image.Image]) -> list[Embedding | None]:
        if not self.is_ready() or not self.load_vision:
            return [None for _ in images]

        if not images:
            return []

        processor = self.processor
        model = self.model
        assert processor is not None
        assert model is not None

        try:
            prepared_images: list[Image.Image] = []

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
                output = self._get_image_features(inputs)

            features = self._normalize_features(output)
            return features.detach().cpu().tolist()
        except Exception as exc:
            logger.error("Error getting image embeddings batch: %s", exc)
            return [None for _ in images]
