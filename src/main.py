"""
Handles ingestion of Steam item data, including image processing and vector database management.
"""
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
from typing import Any, Optional, TypedDict

from PIL import Image
from qdrant_client import QdrantClient, models

from config import settings
from steam_style_embeddings import ColorEmbedder
from utils import download_image, is_animated, is_transparent
from utils.models import get_image_embeddings
from utils.steam_fetcher import SteamFetcher


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class DownloadCandidate(TypedDict):
    item_id: int
    payload: dict[str, Any]
    image_url: str
    has_video: bool
    update_date: datetime


class PendingPoint(TypedDict):
    item_id: int
    payload: dict[str, Any]
    image: Image.Image
    color_vector: list[float]
    update_date: datetime


color_embedder = ColorEmbedder(
    hue_bins=settings.COLOR_HUE_BINS,
    sat_bins=settings.COLOR_SAT_BINS,
    val_bins=settings.COLOR_VAL_BINS,
    sigma_h=settings.COLOR_SIGMA_H,
    sigma_s=settings.COLOR_SIGMA_S,
    sigma_v=settings.COLOR_SIGMA_V,
    power=settings.COLOR_POWER,
)


def main() -> None:
    client: Optional[QdrantClient] = None

    if settings.DATABASE_URL:
        try:
            client = QdrantClient(url=settings.DATABASE_URL)
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
        except Exception as exc:
            logger.warning("Could not initialize Qdrant client: %s", exc)
            client = None
    else:
        logger.warning(
            "Qdrant environment variables missing; skipping vector DB upload.")

    if client is None:
        return

    indexes = [
        ("timestamps.created_at", models.PayloadSchemaType.DATETIME),
        ("timestamps.updated_at", models.PayloadSchemaType.DATETIME),
        ("item.animated", models.PayloadSchemaType.BOOL),
        ("item.transparent", models.PayloadSchemaType.BOOL),
        ("item.tiled", models.PayloadSchemaType.BOOL),
        ("item.id", models.PayloadSchemaType.INTEGER),
        ("item.category", models.PayloadSchemaType.KEYWORD),
    ]

    for field_name, schema in indexes:
        try:
            client.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception as exc:
            logger.warning(
                "Could not create payload index for %s: %s", field_name, exc)

    fetcher = SteamFetcher()
    definitions = fetcher.next_page()
    processed: dict[int, datetime] = {}
    download_workers = max(1, settings.IMAGE_DOWNLOAD_WORKERS)
    pipeline_batch_size = max(
        1,
        min(settings.IMAGE_DOWNLOAD_WORKERS,
            settings.IMAGE_EMBEDDING_BATCH_SIZE),
    )

    while definitions:
        points: list[models.PointStruct] = []
        download_candidates: list[DownloadCandidate] = []

        for definition in definitions:
            try:
                payload = fetcher.map_payload(definition)
                item = payload.get("item", {})
                item_id = item.get("id")

                if item_id is None:
                    continue

                updated_at_value = payload.get(
                    "timestamps", {}).get("updated_at")
                if updated_at_value is None:
                    continue

                try:
                    update_date = datetime.fromisoformat(
                        updated_at_value.replace("Z", "+00:00"))
                except ValueError:
                    logger.warning(
                        "Skipping item %s due to invalid updated_at: %s",
                        item_id,
                        updated_at_value,
                    )
                    continue

                last_processed = processed.get(item_id)
                if last_processed is not None and update_date <= last_processed:
                    continue

                images = item.get("assets", {}).get("images", {})
                image_url = images.get("small") or images.get("large")
                if image_url is None:
                    continue

                videos = item.get("assets", {}).get("videos", {})
                webm = videos.get("webm", {})
                mp4 = videos.get("mp4", {})
                has_video = any([
                    webm.get("large"),
                    webm.get("small"),
                    mp4.get("large"),
                    mp4.get("small"),
                ])

                download_candidates.append(
                    {
                        "item_id": item_id,
                        "payload": payload,
                        "image_url": image_url,
                        "has_video": has_video,
                        "update_date": update_date,
                    }
                )
            except Exception:
                item_id_debug = definition.get("defid", "unknown")
                logger.exception("Error processing item %s", item_id_debug)
                continue

        for start in range(0, len(download_candidates), pipeline_batch_size):
            chunk = download_candidates[start:start + pipeline_batch_size]

            with ThreadPoolExecutor(max_workers=download_workers) as executor:
                downloaded_images = list(
                    executor.map(download_image, [
                                 candidate["image_url"] for candidate in chunk])
                )

            pending_points: list[PendingPoint] = []
            try:
                for candidate, image in zip(chunk, downloaded_images):
                    if image is None:
                        continue

                    item = candidate["payload"].get("item", {})
                    item["animated"] = is_animated(
                        image) or candidate["has_video"]
                    item["transparent"] = is_transparent(image)

                    color_vector = color_embedder.image_to_embedding(
                        image).tolist()
                    pending_points.append(
                        {
                            "item_id": candidate["item_id"],
                            "payload": candidate["payload"],
                            "image": image,
                            "color_vector": color_vector,
                            "update_date": candidate["update_date"],
                        }
                    )

                batch_size = max(1, settings.IMAGE_EMBEDDING_BATCH_SIZE)
                for batch_start in range(0, len(pending_points), batch_size):
                    batch = pending_points[batch_start:batch_start + batch_size]
                    batch_images = [entry["image"] for entry in batch]
                    image_vectors = get_image_embeddings(batch_images)

                    for entry, image_vector in zip(batch, image_vectors):
                        if image_vector is None:
                            continue

                        item_id = entry["item_id"]
                        payload = entry["payload"]
                        color_vector = entry["color_vector"]
                        update_date = entry["update_date"]

                        points.append(
                            models.PointStruct(
                                id=item_id,
                                vector={
                                    "image": image_vector,
                                    "color": color_vector,
                                },
                                payload=payload,
                            )
                        )

                        processed[item_id] = update_date
            finally:
                for image in downloaded_images:
                    if image is not None:
                        image.close()

        if points:
            try:
                client.upload_points(
                    collection_name=settings.COLLECTION_NAME,
                    points=points,
                )
            except (ConnectionError, TimeoutError, ValueError) as exc:
                logger.warning("Failed to upload point to Qdrant: %s", exc)

        definitions = fetcher.next_page()


if __name__ == "__main__":
    main()
