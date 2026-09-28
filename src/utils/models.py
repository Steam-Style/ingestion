import logging

from PIL import Image
from steam_style_embeddings import Embedding, SiglipEmbedder

from config import settings

DEVICE = settings.DEVICE
logger = logging.getLogger(__name__)

siglip_embedder = SiglipEmbedder(
    model_name=settings.MODEL_NAME, device=DEVICE, load_text=False)


def is_model_ready() -> bool:
    return siglip_embedder.is_ready()


def get_image_embeddings(images: list[Image.Image]) -> list[Embedding | None]:
    if not siglip_embedder.is_ready():
        return [None for _ in images]

    try:
        return siglip_embedder.get_image_embeddings(images)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error getting image embeddings batch: %s", e)
        return [None for _ in images]
