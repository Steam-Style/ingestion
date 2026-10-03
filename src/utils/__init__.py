"""
Utility functions for image processing and analysis.
"""
import logging
from io import BytesIO

import av
import requests
from PIL import Image, ImageSequence
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
FRAME_SIZE = 448

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


def download_image(url: str) -> Image.Image | None:
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


def spread_indices(total: int, count: int) -> list[int]:
    """
    Picks up to count indices spread evenly over total items, each from the middle of its share.

    Args:
        total (int): How many items there are.
        count (int): How many to pick.

    Returns:
        list[int]: The picked indices, in order and without repeats.
    """
    if total <= 0 or count <= 0:
        return []

    return sorted({min(total - 1, int((index + 0.5) * total / count)) for index in range(min(count, total))})


def get_video_frames(url: str, count: int) -> list[Image.Image]:
    """
    Downloads a video and returns frames spread evenly over its length.

    Args:
        url (str): The URL of the video.
        count (int): How many frames to return at most.

    Returns:
        list[Image.Image]: The frames, empty if the video could not be downloaded or decoded.
    """
    try:
        response = _session.get(url, timeout=60)
        response.raise_for_status()

        with av.open(BytesIO(response.content)) as container:
            total = sum(1 for _ in container.decode(video=0))

        wanted = set(spread_indices(total, count))
        frames: list[Image.Image] = []

        with av.open(BytesIO(response.content)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    image = frame.to_image()
                    image.thumbnail((FRAME_SIZE, FRAME_SIZE))
                    frames.append(image)
                if len(frames) == len(wanted):
                    break

        return frames

    except requests.RequestException as e:
        logger.warning("Request error downloading video from %s: %s", url, e)
        return []
    except (av.FFmpegError, IndexError, ValueError) as e:
        logger.warning("Error decoding video from %s: %s", url, e)
        return []


def get_image_frames(image: Image.Image, count: int) -> list[Image.Image]:
    """
    Returns frames spread evenly over an animated image, like a GIF or an animated PNG.

    Args:
        image (Image.Image): The animated image.
        count (int): How many frames to return at most.

    Returns:
        list[Image.Image]: The frames, empty if the image has only one.
    """
    frame_count = getattr(image, "n_frames", 1)

    if frame_count <= 1:
        return []

    frames: list[Image.Image] = []

    try:
        for index in spread_indices(frame_count, count):
            image.seek(index)
            frames.append(image.convert("RGBA"))
    except (EOFError, OSError) as e:
        logger.warning("Error reading the frames of an animated image: %s", e)
    finally:
        image.seek(0)

    return frames


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
        try:
            for frame in ImageSequence.Iterator(image):
                alpha = frame.convert("RGBA").getchannel("A")
                extrema = alpha.getextrema()

                if isinstance(extrema, tuple) and len(extrema) == 2:
                    min_alpha = extrema[0]

                    if isinstance(min_alpha, (int, float)) and min_alpha < 255:
                        return True

            return False
        finally:
            image.seek(0)

    alpha = image.convert("RGBA").getchannel("A")
    extrema = alpha.getextrema()

    if isinstance(extrema, tuple) and len(extrema) == 2:
        min_alpha = extrema[0]
        return isinstance(min_alpha, (int, float)) and min_alpha < 255

    return False
