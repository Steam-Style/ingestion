"""
Handles ingestion of Steam item data, including image processing and vector database management.
"""
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

from PIL import Image
from qdrant_client import QdrantClient, models
from steam_style_embeddings import ColorEmbedder

from config import settings
from utils import download_image, is_animated, is_transparent
from utils.models import get_image_embeddings, is_model_ready
from utils.steam_fetcher import SteamFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
heartbeat_file = Path("/tmp/heartbeat")

PAYLOAD_INDEXES = {
    "timestamps.created_at": models.PayloadSchemaType.DATETIME,
    "timestamps.updated_at": models.PayloadSchemaType.DATETIME,
    "item.animated": models.PayloadSchemaType.BOOL,
    "item.transparent": models.PayloadSchemaType.BOOL,
    "item.tiled": models.PayloadSchemaType.BOOL,
    "item.id": models.PayloadSchemaType.INTEGER,
    "item.name": models.PayloadSchemaType.KEYWORD,
    "item.category": models.PayloadSchemaType.KEYWORD,
    "app.id": models.PayloadSchemaType.INTEGER,
}


class DownloadCandidate(TypedDict):
    item_id: int
    payload: dict[str, Any]
    image_url: str


color_embedder = ColorEmbedder(
    hue_bins=settings.COLOR_HUE_BINS,
    sat_bins=settings.COLOR_SAT_BINS,
    val_bins=settings.COLOR_VAL_BINS,
    sigma_h=settings.COLOR_SIGMA_H,
    sigma_s=settings.COLOR_SIGMA_S,
    sigma_v=settings.COLOR_SIGMA_V,
    power=settings.COLOR_POWER,
)


def write_heartbeat() -> None:
    heartbeat_file.write_text(str(time.time()))


def sleep_with_heartbeat(seconds: int) -> None:
    """
    Sleeps for the given number of seconds while keeping the healthcheck heartbeat fresh.
    """
    deadline = time.monotonic() + seconds

    while (remaining := deadline - time.monotonic()) > 0:
        write_heartbeat()
        time.sleep(min(60, remaining))


def ensure_collection(client: QdrantClient) -> None:
    """
    Creates the collection and any payload indexes that do not exist yet.
    """
    if not client.collection_exists(collection_name=settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config={
                "image": models.VectorParams(
                    size=settings.IMAGE_EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
                "color": models.VectorParams(
                    size=color_embedder.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            },
        )

    payload_schema = client.get_collection(
        collection_name=settings.COLLECTION_NAME).payload_schema

    for field_name, schema in PAYLOAD_INDEXES.items():
        if field_name not in payload_schema:
            client.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )


def scroll_points(
    client: QdrantClient,
    scroll_filter: models.Filter | None = None,
    with_payload: bool | list[str] = False,
) -> Iterator[models.Record]:
    """
    Iterates over every point in the collection that matches the filter.
    """
    offset: models.ExtendedPointId | None = None

    while True:
        points, offset = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        yield from points

        if offset is None:
            return


def get_indexed_items(client: QdrantClient) -> dict[int, str | None]:
    """
    Retrieves the Steam update timestamp of every item already stored in the database.
    """
    indexed: dict[int, str | None] = {}

    for point in scroll_points(client, with_payload=["timestamps"]):
        payload = point.payload or {}
        indexed[int(point.id)] = payload.get("timestamps", {}).get("updated_at")

    missing_profiles = models.Filter(
        must=[
            models.FieldCondition(
                key="item.category",
                match=models.MatchValue(value="game profiles"),
            ),
            models.IsEmptyCondition(is_empty=models.PayloadField(key="profile")),
        ]
    )

    for point in scroll_points(client, scroll_filter=missing_profiles):
        indexed.pop(int(point.id), None)

    return indexed


def fix_animated_flags(client: QdrantClient) -> None:
    """
    Marks items that have a video as animated, older versions only checked for the small videos.
    """
    has_large_video = [
        models.Filter(
            must_not=[
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key=f"item.assets.videos.{video_format}.large"))
            ]
        )
        for video_format in ("webm", "mp4")
    ]

    client.set_payload(
        collection_name=settings.COLLECTION_NAME,
        payload={"animated": True},
        key="item",
        points=models.Filter(
            must=[
                models.FieldCondition(
                    key="item.animated",
                    match=models.MatchValue(value=False),
                )
            ],
            should=has_large_video,
        ),
        wait=True,
    )


def get_download_candidates(
    fetcher: SteamFetcher,
    definitions: list[dict[str, Any]],
    indexed: dict[int, str | None],
) -> list[DownloadCandidate]:
    """
    Maps a page of item definitions and keeps the items that are new or were updated since they were indexed.
    """
    candidates: list[DownloadCandidate] = []

    for definition in definitions:
        try:
            payload = fetcher.map_payload(definition)
            item = payload["item"]
            item_id = item["id"]
            updated_at = payload["timestamps"]["updated_at"]

            if item_id is None or updated_at is None:
                continue

            if item_id in indexed and indexed[item_id] == updated_at:
                continue

            images = item["assets"]["images"]
            image_url = images["small"] or images["large"]

            if image_url is None:
                continue

            if item["category"] == "game profiles":
                payload["profile"] = fetcher.get_game_profile(definition)

                background = (payload["profile"] or {}).get("background")
                if background and background["images"]["large"]:
                    image_url = background["images"]["large"]

            candidates.append(
                {
                    "item_id": item_id,
                    "payload": payload,
                    "image_url": image_url,
                }
            )
        except Exception:
            logger.exception("Error processing item %s",
                             definition.get("defid", "unknown"))

    return candidates


def build_points(
    candidates: list[DownloadCandidate],
    images: list[Image.Image | None],
) -> list[models.PointStruct]:
    """
    Computes the color and image embeddings of downloaded images and builds the points to upload.
    """
    prepared: list[tuple[DownloadCandidate, Image.Image, list[float]]] = []

    for candidate, image in zip(candidates, images):
        if image is None:
            continue

        try:
            item = candidate["payload"]["item"]
            item["animated"] = is_animated(image) or item["animated"]
            item["transparent"] = is_transparent(image)
            color_vector = color_embedder.image_to_embedding(image).tolist()
        except Exception:
            logger.exception("Error processing image of item %s",
                             candidate["item_id"])
            continue

        prepared.append((candidate, image, color_vector))

    image_vectors = get_image_embeddings([image for _, image, _ in prepared])

    return [
        models.PointStruct(
            id=candidate["item_id"],
            vector={
                "image": image_vector,
                "color": color_vector,
            },
            payload=candidate["payload"],
        )
        for (candidate, _, color_vector), image_vector in zip(prepared, image_vectors)
        if image_vector is not None
    ]


def ingest_page(
    client: QdrantClient,
    fetcher: SteamFetcher,
    executor: ThreadPoolExecutor,
    definitions: list[dict[str, Any]],
    indexed: dict[int, str | None],
) -> int:
    """
    Downloads, embeds and uploads the new or updated items of a page in small batches.
    """
    candidates = get_download_candidates(fetcher, definitions, indexed)
    batch_size = max(1, settings.IMAGE_EMBEDDING_BATCH_SIZE)
    uploaded = 0

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        images = list(executor.map(
            download_image, [candidate["image_url"] for candidate in batch]))

        try:
            points = build_points(batch, images)
        finally:
            for image in images:
                if image is not None:
                    image.close()

        if points:
            client.upload_points(
                collection_name=settings.COLLECTION_NAME,
                points=points,
                wait=True,
            )
            uploaded += len(points)
            write_heartbeat()

    return uploaded


def run_cycle(client: QdrantClient, fetcher: SteamFetcher, executor: ThreadPoolExecutor) -> None:
    """
    Walks through every page of the points shop once and indexes new or updated items.
    """
    ensure_collection(client)
    indexed = get_indexed_items(client)
    logger.info("Found %d indexed items in the database", len(indexed))

    fetcher.reset()
    uploaded = 0

    while definitions := fetcher.next_page():
        uploaded += ingest_page(client, fetcher, executor,
                                definitions, indexed)
        write_heartbeat()

    logger.info("Ingestion cycle finished, indexed %d new or updated items",
                uploaded)


def main() -> None:
    if not is_model_ready():
        raise SystemExit(f"Could not load embedding model {settings.MODEL_NAME}")

    write_heartbeat()
    client = QdrantClient(url=settings.DATABASE_URL, timeout=60)
    fetcher = SteamFetcher()

    animated_flags_fixed = False

    with ThreadPoolExecutor(max_workers=max(1, settings.IMAGE_DOWNLOAD_WORKERS)) as executor:
        while True:
            try:
                if not animated_flags_fixed:
                    ensure_collection(client)
                    fix_animated_flags(client)
                    animated_flags_fixed = True

                run_cycle(client, fetcher, executor)
                delay = settings.INGESTION_INTERVAL_SECONDS
            except Exception:
                logger.exception("Ingestion cycle failed")
                delay = settings.INGESTION_RETRY_SECONDS

            sleep_with_heartbeat(delay)


if __name__ == "__main__":
    main()
