import logging

from config import settings
from typing import Optional, List
from PIL import Image
from steam_style_embeddings import SiglipEmbedder

DEVICE = settings.DEVICE
logger = logging.getLogger(__name__)

siglip_embedder = SiglipEmbedder(model_name=settings.MODEL_NAME, device=DEVICE)

Embedding = List[float]


def get_text_embedding(text: str) -> Optional[Embedding]:
    if not siglip_embedder.is_ready():
        return None

    try:
        return siglip_embedder.get_text_embedding(text)
    except Exception as e:
        logger.error("Error getting text embedding: %s", e)
        return None


def get_image_embedding(image: Image.Image) -> Optional[Embedding]:
    if not siglip_embedder.is_ready():
        return None

    try:
        return siglip_embedder.get_image_embedding(image)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error getting image embedding: %s", e)
        return None
