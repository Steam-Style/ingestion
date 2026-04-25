"""
Module to interact with the points shop API.
"""
from datetime import datetime, timezone
from typing import Any, Optional
from steam.client import SteamClient  # type: ignore

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE_URL = "https://api.steampowered.com/ILoyaltyRewardsService/QueryRewardItems/v1"
IMAGE_BASE_URLS = [
    "https://cdn.akamai.steamstatic.com/steamcommunity/public/images/items",
    "https://shared.fastly.steamstatic.com/community_assets/images/items"
]
ICON_BASE_URL = "https://shared.fastly.steamstatic.com/community_assets/images/apps"
CATEGORIES = {
    0: "item bundles",
    1: "badge collections",
    3: "profile backgrounds",
    4: "emoticons",
    8: "game profiles",
    11: "animated stickers",
    12: "chat effects",
    13: "mini-profile backgrounds",
    14: "avatar frames",
    15: "avatars",
    16: "steam deck keyboards",
    17: "steam startup movies",
}


class SteamFetcher:
    """
    Fetches Steam item data from the points shop API.
    """

    def __init__(self) -> None:
        self.current_response: Optional[dict[str, Any]] = None
        self.total_count: Optional[int] = None
        self.session = requests.Session()
        retry_policy = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods={"GET"},
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_policy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.apps: dict[Any, dict[str, Any]] = {}

        client = SteamClient()
        client.anonymous_login()
        self.client = client

    def next_page(self) -> Optional[list[dict[str, Any]]]:
        """
        Fetches the next page of item definitions from the Steam Points Shop API.

        Returns:
            Optional[list[dict]]: A list of item definitions, or None if no more pages.
        """
        cursor = None

        if self.current_response is not None:
            cursor = self.current_response.get("next_cursor")

        if cursor is None:
            self.apps = {}

        try:
            response = self.session.get(
                API_BASE_URL,
                params={
                    "cursor": cursor,
                    "count": 1000,
                },
                timeout=30,
            )

            response.raise_for_status()
            data = response.json()
            response_data = data.get("response", {})
            definitions = response_data.get("definitions", [])
            self.total_count = response_data.get("total_count", None)
        except (requests.RequestException, ValueError) as exc:
            print(f"Failed to fetch Steam Points Shop page: {exc}")
            return None

        if definitions is not None and len(definitions) > 0:
            self.current_response = response_data
            return definitions

        return None

    def _get_app_info(self, app_id: Optional[int]) -> Optional[dict[str, Any]]:
        """
        Retrieves application info, fetching it if not cached.

        Args:
            app_id (Optional[int]): The application ID.

        Returns:
            Optional[dict[str, Any]]: The application info dictionary, or None.
        """
        if app_id is None:
            return None

        if app_id in self.apps:
            return self.apps[app_id]

        try:
            product_info: Optional[dict[str, Any]] = self.client.get_product_info(  # type: ignore
                [app_id])
            if product_info and "apps" in product_info:
                app_info = product_info["apps"].get(app_id)
                if app_info:
                    self.apps[app_id] = app_info
                    return app_info
        except BaseException:
            pass

        return None

    def _generate_asset_url(self, app_id: Optional[int], path: Optional[str]) -> Optional[str]:
        """
        Generates a full URL for a Steam asset path.

        Args:
            app_id (Optional[int]): The application ID.
            path (Optional[str]): The asset path.

        Returns:
            Optional[str]: The full URL, or None.
        """
        if path and app_id:
            base_url = IMAGE_BASE_URLS[1] if "/" in path else IMAGE_BASE_URLS[0]
            return f"{base_url}/{app_id}/{path}"
        return None

    def _parse_timestamp(self, timestamp: Optional[int]) -> Optional[str]:
        """
        Converts a unix timestamp to a UTC ISO-8601 string.

        Args:
            timestamp (Optional[int]): The unix timestamp.

        Returns:
            Optional[str]: UTC timestamp in ISO-8601 format, or None.
        """
        if not timestamp:
            return None

        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    def map_payload(self, definition: dict[str, Any]) -> dict[str, Any]:
        """
        Maps a raw API result to a more structured schema.

        Args:
            definition (dict[str, Any]): The raw item definition from the API.

        Returns:
            dict[str, Any]: The mapped item payload.
        """
        app_id = definition.get("appid")
        item_id = definition.get("defid")
        community_item_data = definition.get("community_item_data", {})

        app_info = self._get_app_info(app_id)
        common_info: dict[str, Any] = app_info.get(
            "common", {}) if app_info else {}

        app_name = common_info.get("name")
        app_icon_path = common_info.get("icon")
        app_icon_url = f"{ICON_BASE_URL}/{app_id}/{app_icon_path}.jpg" if app_icon_path and app_id else None

        def get_url(key: str) -> Optional[str]:
            return self._generate_asset_url(app_id, community_item_data.get(key))

        small_image_url = get_url("item_image_small")

        has_video = any(get_url(key) for key in [
            "item_movie_webm_large", "item_movie_webm_small",
            "item_movie_mp4_large", "item_movie_mp4_small"
        ])

        item_animated = has_video
        item_transparent = False
        item_tiled = community_item_data.get("tiled", False)

        name = definition.get("name") or community_item_data.get("item_name")
        item_class = definition.get("community_item_class")
        category = CATEGORIES.get(item_class, str(item_class)) if isinstance(
            item_class, int) else "Unknown"

        market_item_url = None
        market_app_url = None
        shop_item_url = None
        shop_app_url = None

        if app_id:
            safe_name = str(name).replace(" ", "%20") if name else ""
            market_item_url = f"https://steamcommunity.com/market/listings/753/{app_id}-{safe_name}"
            market_app_url = f"https://steamcommunity.com/market/search?appid=753&category_753_Game%5B%5D=tag_app_{app_id}"
            shop_app_url = f"https://store.steampowered.com/points/shop/app/{app_id}"

        if item_id:
            shop_item_url = f"https://store.steampowered.com/points/shop/reward/{item_id}"

        return {
            "item": {
                "id": item_id,
                "name": name,
                "title": community_item_data.get("item_title"),
                "description": community_item_data.get("item_description"),
                "internal_description": definition.get("internal_description"),
                "category": category,
                "point_cost": definition.get("point_cost"),
                "animated": item_animated,
                "transparent": item_transparent,
                "tiled": item_tiled,
                "assets": {
                    "images": {
                        "large": get_url("item_image_large"),
                        "small": small_image_url,
                    },
                    "videos": {
                        "webm": {
                            "large": get_url("item_movie_webm"),
                            "small": get_url("item_movie_webm_small"),
                        },
                        "mp4": {
                            "large": get_url("item_movie_mp4"),
                            "small": get_url("item_movie_mp4_small"),
                        }
                    }
                },
            },
            "app": {
                "id": app_id,
                "name": app_name,
                "icon": app_icon_url
            },
            "urls": {
                "market": {
                    "item": market_item_url,
                    "app": market_app_url,
                },
                "points_shop": {
                    "item": shop_item_url,
                    "app": shop_app_url,
                },
            },
            "timestamps": {
                "created_at": self._parse_timestamp(definition.get("timestamp_created")),
                "updated_at": self._parse_timestamp(definition.get("timestamp_updated")),
                "available_at": self._parse_timestamp(definition.get("timestamp_available")),
                "unavailable_at": self._parse_timestamp(definition.get("timestamp_unavailable")),
                "usable_duration_seconds": definition.get("usable_duration"),
            },
        }
