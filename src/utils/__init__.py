"""
Utility functions for image processing and analysis.
"""
import logging
from typing import Optional, List
from io import BytesIO

from PIL import Image
import requests


Embedding = List[float]
logger = logging.getLogger(__name__)


def download_image(url: str) -> Optional[Image.Image]:
    """
    Downloads an image from a URL and returns it as a PIL Image.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Optional[Image.Image]: The downloaded image as a PIL Image, or None if the download fails.
    """
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        logger.warning("Request error downloading image from %s: %s", url, e)
        return None
    except (Image.UnidentifiedImageError, OSError) as e:
        logger.warning("Error decoding image from %s: %s", url, e)
        return None


def is_animated(image: Image.Image) -> bool:
    """
    Checks if a given PIL Image is animated (i.e., has multiple frames).

    Args:
        image (Image.Image): The PIL Image to check.

    Returns:
        bool: True if the image is animated, False otherwise.
    """
    return getattr(image, "is_animated", False)


def is_transparent(image: Image.Image) -> bool:
    """
    Checks if a given PIL Image has transparency.

    Args:
        image (Image.Image): The PIL Image to check.

    Returns:
        bool: True if the image has transparency, False otherwise.
    """
    if "transparency" in image.info:
        return True
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        return True
    return False
