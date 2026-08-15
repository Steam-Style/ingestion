import logging

from PIL import Image
from steam_style_embeddings import SiglipEmbedder

from config import settings

DEVICE = settings.DEVICE
logger = logging.getLogger(__name__)

siglip_embedder = SiglipEmbedder(model_name=settings.MODEL_NAME, device=DEVICE)

Embedding = list[float]


def get_text_embedding(text: str) -> Embedding | None:
    if not siglip_embedder.is_ready():
        return None

    try:
        return siglip_embedder.get_text_embedding(text)
    except Exception as e:
        logger.error("Error getting text embedding: %s", e)
        return None


def get_image_embedding(image: Image.Image) -> Embedding | None:
    if not siglip_embedder.is_ready():
        return None

    try:
        return siglip_embedder.get_image_embedding(image)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error getting image embedding: %s", e)
        return None


def get_image_embeddings(images: list[Image.Image]) -> list[Embedding | None]:
    if not siglip_embedder.is_ready():
        return [None for _ in images]

    try:
        return siglip_embedder.get_image_embeddings(images)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error getting image embeddings batch: %s", e)
        return [None for _ in images]
