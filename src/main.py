"""
Handles ingestion of Steam item data, including image processing and vector database management.
"""
import logging
from typing import Optional

from tqdm import tqdm
from qdrant_client import QdrantClient, models
import pillow_avif

from utils.color_embed import ColorEmbedder
from config import settings
from utils.models import get_image_embedding
from utils import download_image
from utils.steam_fetcher import SteamFetcher


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


Embedder = ColorEmbedder(
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
                            size=Embedder.embedding_dimension,
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
        processed: dict[int, int] = {}

        while definitions:
            points: list[models.PointStruct] = []

            for definition in definitions:
                try:
                    payload = fetcher.map_payload(definition)
                    item = payload.get("item", {})
                    item_id = item.get("id")
                    update_date = payload.get(
                        "timestamps", {}).get("updated_at")
                    last_processed = processed.get(item_id)

                    if item_id is None or last_processed is None or update_date <= last_processed:
                        continue

                    images = item.get("assets", {}).get("images", {})
                    image_url = images.get("large") or images.get("small")

                    if image_url is None:
                        continue

                    image = download_image(image_url)

                    if image is None:
                        continue

                    image_vector = get_image_embedding(image)
                    color_vector = Embedder.image_to_embedding(image).tolist()

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
                except BaseException:
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
