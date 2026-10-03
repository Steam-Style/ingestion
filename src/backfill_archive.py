"""
One-time backfill of points shop items that Steam no longer lists, using the Points Shop API responses the Internet
Archive saved since 2020. Items that are found are indexed like any other item and marked as unavailable. Only items
that show up in an archived response can be found, so some will still be missing.

Run it where the ingestion runs, so it reaches the database and has the embedding model:

    python src/backfill_archive.py --dry-run
    python src/backfill_archive.py

Downloaded responses are kept in the cache directory, so the script can be stopped and started again without fetching
them twice.
"""
import argparse
import base64
import hashlib
import json
import logging
import time
import urllib.parse
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from qdrant_client import QdrantClient

from config import settings
from main import ensure_collection, ingest_page, scroll_points
from utils.steam_fetcher import API_BASE_URL, CATEGORIES, SteamFetcher

logger = logging.getLogger("backfill")

CDX_URL = "https://web.archive.org/cdx/search/cdx"
ARCHIVED_ENDPOINT = "api.steampowered.com/ILoyaltyRewardsService/BatchedQueryRewardItems*"
SNAPSHOT_URL = "https://web.archive.org/web/{timestamp}id_/{original}"
PAGE_SIZE = 100

DEFINITION_FIELDS: dict[int, tuple[str, str]] = {
    1: ("appid", "int"),
    2: ("defid", "int"),
    3: ("type", "int"),
    4: ("community_item_class", "int"),
    5: ("community_item_type", "int"),
    6: ("point_cost", "str_int"),
    7: ("timestamp_created", "int"),
    8: ("timestamp_updated", "int"),
    9: ("timestamp_available", "int"),
    10: ("quantity", "str_int"),
    11: ("internal_description", "str"),
    12: ("active", "bool"),
    14: ("timestamp_available_end", "int"),
    16: ("usable_duration", "int"),
    17: ("bundle_discount", "int"),
}

ITEM_DATA_FIELDS: dict[int, tuple[str, str]] = {
    1: ("item_name", "str"),
    2: ("item_title", "str"),
    3: ("item_description", "str"),
    4: ("item_image_small", "str"),
    5: ("item_image_large", "str"),
    6: ("item_movie_webm", "str"),
    7: ("item_movie_mp4", "str"),
    8: ("animated", "bool"),
    10: ("item_movie_webm_small", "str"),
    11: ("item_movie_mp4_small", "str"),
    12: ("profile_theme_id", "str"),
    13: ("tiled", "bool"),
}


def read_varint(data: bytes, position: int) -> tuple[int, int]:
    """
    Reads a protobuf varint and returns it with the position after it.
    """
    result = 0
    shift = 0

    while True:
        byte = data[position]
        position += 1
        result |= (byte & 0x7F) << shift

        if not byte & 0x80:
            return result, position

        shift += 7


def read_fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """
    Yields the field number, wire type and raw value of every field in a protobuf message.
    """
    position = 0

    while position < len(data):
        key, position = read_varint(data, position)
        number, wire_type = key >> 3, key & 7

        if wire_type == 0:
            value, position = read_varint(data, position)
            yield number, wire_type, value
        elif wire_type == 2:
            length, position = read_varint(data, position)
            yield number, wire_type, data[position:position + length]
            position += length
        elif wire_type == 1:
            yield number, wire_type, data[position:position + 8]
            position += 8
        elif wire_type == 5:
            yield number, wire_type, data[position:position + 4]
            position += 4
        else:
            raise ValueError(f"Unsupported wire type {wire_type}")


def decode_message(data: bytes, layout: dict[int, tuple[str, str]]) -> dict[str, Any]:
    """
    Decodes the fields of a protobuf message that are in the layout, in the same shape as the JSON API.
    """
    message: dict[str, Any] = {}

    for number, _, value in read_fields(data):
        if number not in layout:
            continue

        name, kind = layout[number]

        if kind == "str" and isinstance(value, bytes):
            message[name] = value.decode("utf-8", errors="replace")
        elif kind == "int" and isinstance(value, int):
            message[name] = value
        elif kind == "str_int" and isinstance(value, int):
            message[name] = str(value)
        elif kind == "bool" and isinstance(value, int):
            message[name] = bool(value)

    return message


def decode_definition(data: bytes) -> dict[str, Any]:
    """
    Decodes a LoyaltyRewardDefinition into the shape the JSON API returns.
    """
    definition = decode_message(data, DEFINITION_FIELDS)
    bundle_ids: list[int] = []

    for number, wire_type, value in read_fields(data):
        if number == 13 and isinstance(value, bytes):
            definition["community_item_data"] = decode_message(value, ITEM_DATA_FIELDS)
        elif number == 15 and wire_type == 0 and isinstance(value, int):
            bundle_ids.append(value)
        elif number == 15 and isinstance(value, bytes):
            position = 0
            while position < len(value):
                bundle_id, position = read_varint(value, position)
                bundle_ids.append(bundle_id)

    if bundle_ids:
        definition["bundle_defids"] = bundle_ids

    return definition


def decode_response(content: bytes) -> list[dict[str, Any]]:
    """
    Reads the item definitions from an archived BatchedQueryRewardItems response, protobuf or JSON.
    """
    if content.lstrip().startswith(b"{"):
        data = json.loads(content)
        return [
            definition
            for entry in data.get("response", {}).get("responses", [])
            for definition in entry.get("response", {}).get("definitions", [])
        ]

    definitions: list[dict[str, Any]] = []

    for number, _, entry in read_fields(content):
        if number != 1 or not isinstance(entry, bytes):
            continue
        for entry_number, _, body in read_fields(entry):
            if entry_number != 2 or not isinstance(body, bytes):
                continue
            for body_number, _, definition in read_fields(body):
                if body_number == 1 and isinstance(definition, bytes):
                    definitions.append(decode_definition(definition))

    return definitions


def describe_request(original: str) -> tuple[set[int], str | None]:
    """
    Reads which item IDs and language an archived request asked for. Requests without IDs list shop pages.
    """
    query = urllib.parse.parse_qs(urllib.parse.urlparse(original).query)
    item_ids: set[int] = set()
    language: str | None = None

    if "input_protobuf_encoded" in query:
        raw = base64.b64decode(query["input_protobuf_encoded"][0] + "==")
        for number, _, request in read_fields(raw):
            if number != 1 or not isinstance(request, bytes):
                continue
            for field, wire_type, value in read_fields(request):
                if field == 4 and isinstance(value, bytes):
                    language = value.decode("utf-8", errors="replace")
                elif field == 11 and wire_type == 0 and isinstance(value, int):
                    item_ids.add(value)
                elif field == 11 and isinstance(value, bytes):
                    position = 0
                    while position < len(value):
                        item_id, position = read_varint(value, position)
                        item_ids.add(item_id)

    if "input_json" in query:
        body = json.loads(query["input_json"][0])
        for request in body.get("requests", []):
            item_ids.update(int(item_id) for item_id in request.get("definitionids", []))
            language = request.get("language", language)

    return item_ids, language


def get_snapshots(session: requests.Session) -> list[tuple[str, str]]:
    """
    Lists one archived snapshot per distinct request to the Points Shop API.
    """
    response = session.get(
        CDX_URL,
        params={
            "url": ARCHIVED_ENDPOINT,
            "output": "json",
            "fl": "timestamp,original",
            "filter": "statuscode:200",
            "collapse": "urlkey",
        },
        timeout=600,
    )
    response.raise_for_status()
    return [(timestamp, original) for timestamp, original in response.json()[1:]]


def get_listed_ids(session: requests.Session) -> set[int]:
    """
    Retrieves the IDs of every item the points shop has right now, including ones that aren't sold on their own.
    """
    listed: set[int] = set()
    cursor: str | None = None

    while True:
        response = session.get(
            API_BASE_URL,
            params={"count": 1000, "cursor": cursor, "include_direct_purchase_disabled": "true"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json().get("response", {})
        definitions = data.get("definitions", [])
        listed.update(int(definition["defid"]) for definition in definitions)

        if not definitions or data.get("next_cursor") in (None, cursor):
            return listed

        cursor = data["next_cursor"]


def get_stored_ids(client: QdrantClient) -> set[int]:
    """
    Retrieves the IDs of every item in the database.
    """
    return {int(point.id) for point in scroll_points(client)}


def download_snapshot(session: requests.Session, cache_dir: Path, timestamp: str, original: str) -> bytes | None:
    """
    Downloads an archived response, or reads it from the cache when it was downloaded before.
    """
    path = cache_dir / f"{hashlib.sha1(original.encode()).hexdigest()}.bin"

    if path.exists():
        return path.read_bytes()

    for attempt in range(5):
        try:
            response = session.get(SNAPSHOT_URL.format(timestamp=timestamp, original=original), timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue

        if response.status_code == 200:
            path.write_bytes(response.content)
            return response.content

        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(15 * (attempt + 1))
            continue

        return None

    return None


def collect_definitions(
    session: requests.Session,
    cache_dir: Path,
    known: set[int],
    workers: int,
) -> dict[int, dict[str, Any]]:
    """
    Gathers the archived definitions of items that aren't known yet, preferring English ones.
    """
    snapshots = get_snapshots(session)
    selected: list[tuple[str, str]] = []
    covered: set[int] = set()

    described = [(timestamp, original, *describe_request(original)) for timestamp, original in snapshots]
    described.sort(key=lambda entry: entry[3] != "english")

    for timestamp, original, item_ids, _ in described:
        if not item_ids:
            selected.append((timestamp, original))
        elif item_ids - known - covered:
            selected.append((timestamp, original))
            covered.update(item_ids)

    logger.info("%d archived requests, downloading %d of them", len(snapshots), len(selected))

    found: dict[int, dict[str, Any]] = {}
    languages = {original: language for _, original, _, language in described}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        contents = pool.map(lambda snapshot: (snapshot[1], download_snapshot(session, cache_dir, *snapshot)), selected)

        for index, (original, content) in enumerate(contents, start=1):
            if index % 500 == 0:
                logger.info("Read %d of %d archived responses, %d new items so far", index, len(selected), len(found))

            if not content:
                continue

            try:
                definitions = decode_response(content)
            except (ValueError, IndexError, json.JSONDecodeError):
                logger.warning("Could not read the archived response of %s", original)
                continue

            english = languages.get(original) in (None, "english")

            for definition in definitions:
                item_id = definition.get("defid")
                if not isinstance(item_id, int) or item_id in known:
                    continue
                if definition.get("community_item_class") not in CATEGORIES:
                    continue
                if item_id not in found or english:
                    found[item_id] = definition

    return found


def mark_unavailable(client: QdrantClient, item_ids: list[int]) -> None:
    """
    Marks backfilled items as no longer sold. When they were removed isn't known, so that stays empty.
    """
    stored = [
        int(point.id)
        for start in range(0, len(item_ids), 1000)
        for point in client.retrieve(
            collection_name=settings.COLLECTION_NAME,
            ids=item_ids[start:start + 1000],
            with_payload=False,
            with_vectors=False,
        )
    ]

    for start in range(0, len(stored), 1000):
        client.set_payload(
            collection_name=settings.COLLECTION_NAME,
            payload={"available": False},
            key="item",
            points=stored[start:start + 1000],
            wait=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="archive_cache", help="Where downloaded responses are kept")
    parser.add_argument("--workers", type=int, default=2, help="Archive downloads at the same time")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be added")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Steam Style backfill (https://steam.style)"})

    client = QdrantClient(url=settings.DATABASE_URL, timeout=60)
    ensure_collection(client)

    stored = get_stored_ids(client)
    listed = get_listed_ids(session)
    logger.info("%d items in the database, %d listed in the points shop", len(stored), len(listed))

    found = collect_definitions(session, cache_dir, stored | listed, args.workers)
    by_category: dict[str, int] = {}
    for definition in found.values():
        category = CATEGORIES.get(definition.get("community_item_class"), "unknown")
        by_category[category] = by_category.get(category, 0) + 1

    logger.info("Found %d items that are no longer sold: %s", len(found), by_category)

    if args.dry_run or not found:
        return

    fetcher = SteamFetcher()
    definitions = list(found.values())
    uploaded = 0

    with ThreadPoolExecutor(max_workers=max(1, settings.IMAGE_DOWNLOAD_WORKERS)) as executor:
        for start in range(0, len(definitions), PAGE_SIZE):
            page = definitions[start:start + PAGE_SIZE]
            fetcher._prefetch_app_info(page)
            uploaded += ingest_page(client, fetcher, executor, page, {}, set())
            mark_unavailable(client, [definition["defid"] for definition in page])
            logger.info("Indexed %d of %d items", min(start + PAGE_SIZE, len(definitions)), len(definitions))

    logger.info("Backfill finished, added %d items", uploaded)


if __name__ == "__main__":
    main()
