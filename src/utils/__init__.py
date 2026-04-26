"""
Utility functions for image processing and analysis.
"""
import logging
from typing import Optional, List
from io import BytesIO

from PIL import Image, ImageSequence
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


Embedding = List[float]
logger = logging.getLogger(__name__)

_session = requests.Session()
_retry_policy = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods={"GET"},
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry_policy,
                       pool_connections=32, pool_maxsize=64)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def download_image(url: str) -> Optional[Image.Image]:
    """
    Downloads an image from a URL and returns it as a PIL Image.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Optional[Image.Image]: The downloaded image as a PIL Image, or None if the download fails.
    """
    try:
        response = _session.get(url, timeout=20)
        response.raise_for_status()

        buffer = BytesIO(response.content)
        image = Image.open(buffer)
        image._buffer = buffer  # type: ignore[attr-defined]
        return image

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
    frame_count = getattr(image, "n_frames", 1)
    return bool(getattr(image, "is_animated", False) or frame_count > 1)


def is_transparent(image: Image.Image) -> bool:
    """
    Checks if a given PIL Image has transparency.

    Args:
        image (Image.Image): The PIL Image to check.

    Returns:
        bool: True if the image has transparency, False otherwise.
    """
    if is_animated(image):
        for frame in ImageSequence.Iterator(image):
            alpha = frame.convert("RGBA").getchannel("A")
            extrema = alpha.getextrema()

            if isinstance(extrema, tuple) and len(extrema) == 2:
                min_alpha = extrema[0]

                if isinstance(min_alpha, (int, float)) and min_alpha < 255:
                    return True

        return False

    alpha = image.convert("RGBA").getchannel("A")
    extrema = alpha.getextrema()

    if isinstance(extrema, tuple) and len(extrema) == 2:
        min_alpha = extrema[0]
        return isinstance(min_alpha, (int, float)) and min_alpha < 255

    return False
