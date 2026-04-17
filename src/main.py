"""
Handles ingestion of Steam item data, including image processing and vector database management.
"""
import logging
from datetime import datetime
from typing import Optional
from qdrant_client import QdrantClient, models
from steam_style_embeddings import ColorEmbedder
from config import settings
from utils.models import get_image_embedding
from utils import download_image
from utils.steam_fetcher import SteamFetcher


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


color_embedder = ColorEmbedder(
    hue_bins=settings.COLOR_HUE_BINS,
    sat_bins=settings.COLOR_SAT_BINS,
    val_bins=settings.COLOR_VAL_BINS,
    sigma_h=settings.COLOR_SIGMA_H,
    sigma_s=settings.COLOR_SIGMA_S,
    sigma_v=settings.COLOR_SIGMA_V,
    power=settings.COLOR_POWER,
)

if __name__ == "__main__":
    client: Optional[QdrantClient] = None

    if settings.DATABASE_URL:
        try:
            client = QdrantClient(
                url=settings.DATABASE_URL,
            )
            if not client.collection_exists(collection_name=settings.COLLECTION_NAME):
                client.create_collection(
                    collection_name=settings.COLLECTION_NAME,
                    vectors_config={
                        "image": models.VectorParams(
                            size=settings.IMAGE_EMBEDDING_DIM,
                            distance=models.Distance.COSINE
                        ),
                        "color": models.VectorParams(
                            size=color_embedder.embedding_dimension,
                            distance=models.Distance.COSINE
                        )
                    }
                )
        except Exception as e:
            logger.warning("Could not initialize Qdrant client: %s", e)
            client = None
    else:
        logger.warning(
            "Qdrant environment variables missing; skipping vector DB upload.")

    if client is not None:
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
            except Exception as e:
                logger.warning(
                    "Could not create payload index for %s: %s", field_name, e)

        fetcher = SteamFetcher()
        definitions = fetcher.next_page()
        processed: dict[int, datetime] = {}

        while definitions:
            points: list[models.PointStruct] = []

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
                            "Skipping item %s due to invalid updated_at: %s", item_id, updated_at_value)
                        continue

                    last_processed = processed.get(item_id)

                    if last_processed is not None and update_date <= last_processed:
                        continue

                    images = item.get("assets", {}).get("images", {})
                    image_url = images.get("large") or images.get("small")

                    if image_url is None:
                        continue

                    image = download_image(image_url)

                    if image is None:
                        continue

                    image_vector = get_image_embedding(image)
                    color_vector = color_embedder.image_to_embedding(
                        image).tolist()

                    if image_vector is None:
                        continue

                    points.append(
                        models.PointStruct(
                            id=item_id,
                            vector={
                                "image": image_vector,
                                "color": color_vector
                            },
                            payload=payload
                        )
                    )

                    processed[item_id] = update_date
                except Exception:
                    item_id_debug = definition.get("defid", "unknown")
                    logger.exception("Error processing item %s", item_id_debug)
                    continue

            if points:
                try:
                    client.upload_points(
                        collection_name=settings.COLLECTION_NAME,
                        points=points
                    )
                except (ConnectionError, TimeoutError, ValueError) as e:
                    logger.warning("Failed to upload point to Qdrant: %s", e)

            definitions = fetcher.next_page()
