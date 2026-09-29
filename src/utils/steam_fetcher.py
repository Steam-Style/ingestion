"""
Module to interact with the points shop API.
"""
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import requests
from gevent import Timeout
from requests import utils
from requests.adapters import HTTPAdapter
from steam.client import SteamClient
from urllib3.util.retry import Retry

from config import settings

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.steampowered.com/ILoyaltyRewardsService/QueryRewardItems/v1"
CDN_BASE_URL = "https://shared.fastly.steamstatic.com/community_assets/images"
PROFILE_THEMES_CSS_URL = "https://community.fastly.steamstatic.com/public/css/skin_1/profilev2.css"
PROFILE_PREVIEW_INTERVAL_SECONDS = 5
PROFILE_PREVIEW_ATTEMPTS = 3
PROFILE_PREVIEW_BACKOFF_SECONDS = 60
PROFILE_PREVIEW_COOLDOWN_SECONDS = 900
BUNDLE_PARTS = {
    3: "background",
    13: "mini_profile",
    15: "avatar",
    14: "frame",
}
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
        self.current_response: dict[str, Any] | None = None
        self.total_count: int | None = None
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
        self.named_themes: dict[str, dict[str, str]] | None = None
        self.last_preview_request = 0.0
        self.previews_paused_until = 0.0

        default_ua = utils.default_user_agent()
        custom_ua = f"{default_ua} (Steam-Style/1.0)"
        self.session.headers.update({"User-Agent": custom_ua})

        client = SteamClient()
        client.anonymous_login()
        self.client = client

    def _prefetch_app_info(self, definitions: list[dict[str, Any]]) -> None:
        """
        Fetches app metadata for all uncached app IDs in the page with one request.
        """
        app_ids = {
            app_id
            for definition in definitions
            for app_id in [definition.get("appid")]
            if isinstance(app_id, int) and app_id not in self.apps
        }

        if not app_ids:
            return

        for _ in range(3):
            try:
                product_info: dict[str, Any] | None = self.client.get_product_info(
                    list(app_ids)
                )

                if not product_info or "apps" not in product_info:
                    continue

                apps_data = product_info.get("apps", {})

                if not apps_data:
                    continue

                for app_id in app_ids:
                    app_info = apps_data.get(app_id)

                    if app_info:
                        self.apps[app_id] = app_info

                return

            except (Exception, Timeout):
                self.client.anonymous_login()

    def reset(self) -> None:
        """
        Restarts pagination from the first page and clears the cached app metadata.
        """
        self.current_response = None
        self.apps = {}
        self.named_themes = None

    def next_page(self) -> list[dict[str, Any]]:
        """
        Fetches the next page of item definitions from the Steam Points Shop API.

        Returns:
            list[dict[str, Any]]: A list of item definitions, empty once every page has been fetched.

        Raises:
            requests.RequestException: If the page could not be fetched.
            ValueError: If the response is not valid JSON.
        """
        cursor = None

        if self.current_response is not None:
            cursor = self.current_response.get("next_cursor")

        response = self.session.get(
            API_BASE_URL,
            params={
                "cursor": cursor,
                "count": 1000,
            },
            timeout=30,
        )

        response.raise_for_status()
        response_data = response.json().get("response", {})
        definitions = response_data.get("definitions", [])
        self.total_count = response_data.get("total_count", None)
        self.current_response = response_data

        if definitions:
            self._prefetch_app_info(definitions)

        return definitions

    def _get_app_info(self, app_id: int | None) -> dict[str, Any] | None:
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
            product_info: dict[str, Any] | None = self.client.get_product_info(
                [app_id])

            if product_info and "apps" in product_info:
                app_info = product_info["apps"].get(app_id)
                if app_info:
                    self.apps[app_id] = app_info
                    return app_info

        except (Exception, Timeout):
            pass

        return None

    def _generate_asset_url(self, app_id: int | None, path: str | None) -> str | None:
        """
        Generates a full URL for a Steam asset path.

        Args:
            app_id (Optional[int]): The application ID.
            path (Optional[str]): The asset path.

        Returns:
            Optional[str]: The full URL, or None.
        """
        if path and app_id:
            return f"{CDN_BASE_URL}/items/{app_id}/{path}"

        return None

    def _map_assets(self, app_id: int | None, data: dict[str, Any]) -> dict[str, Any]:
        """
        Maps the image and video paths of an item definition to full URLs.

        Args:
            app_id (Optional[int]): The application ID.
            data (dict[str, Any]): The community item data of the definition.

        Returns:
            dict[str, Any]: The images and videos of the item.
        """
        def get_url(key: str) -> str | None:
            return self._generate_asset_url(app_id, data.get(key))

        return {
            "images": {
                "large": get_url("item_image_large"),
                "small": get_url("item_image_small"),
            },
            "videos": {
                "webm": {
                    "large": get_url("item_movie_webm"),
                    "small": get_url("item_movie_webm_small"),
                },
                "mp4": {
                    "large": get_url("item_movie_mp4"),
                    "small": get_url("item_movie_mp4_small"),
                },
            },
        }

    def _parse_timestamp(self, timestamp: int | None) -> str | None:
        """
        Converts a unix timestamp to a UTC ISO-8601 string.

        Args:
            timestamp (Optional[int]): The unix timestamp.

        Returns:
            Optional[str]: UTC timestamp in ISO-8601 format, or None.
        """
        if not timestamp:
            return None

        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")

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
        app_icon_url = f"{CDN_BASE_URL}/apps/{app_id}/{app_icon_path}.jpg" if app_icon_path and app_id else None

        assets = self._map_assets(app_id, community_item_data)
        has_video = any(
            url for video in assets["videos"].values() for url in video.values())

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
                "assets": assets,
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
                "unavailable_at": self._parse_timestamp(definition.get("timestamp_available_end")),
                "usable_duration_seconds": definition.get("usable_duration"),
            },
        }

    def _parse_theme_colors(self, css: str) -> dict[str, str]:
        """
        Parses the CSS variables of a profile theme rule into a dictionary.

        Args:
            css (str): The body of a theme's CSS rule.

        Returns:
            dict[str, str]: The colors, keyed by variable name without the leading dashes.
        """
        return {
            name: value.strip()
            for name, value in re.findall(r"--([a-z0-9-]+)\s*:\s*([^;]+);", css)
        }

    def _get_named_theme(self, theme_id: str) -> dict[str, str] | None:
        """
        Retrieves the colors of one of Steam's named profile themes from the profile stylesheet.

        Args:
            theme_id (str): The theme ID, such as "Midnight".

        Returns:
            Optional[dict[str, str]]: The theme colors, or None if the theme is unknown.
        """
        if self.named_themes is None:
            response = self.session.get(PROFILE_THEMES_CSS_URL, timeout=30)
            response.raise_for_status()
            self.named_themes = {
                name: self._parse_theme_colors(body)
                for name, body in re.findall(r"body\.(\w+?)Theme\s*\{([^}]*)\}", response.text)
            }

        return self.named_themes.get(theme_id)

    def _get_game_profile_theme(self, app_id: int | None, item_type: int | None) -> dict[str, str] | None:
        """
        Retrieves the colors of a game specific profile theme, which Steam only includes on profile pages.

        Args:
            app_id (Optional[int]): The application ID of the game profile.
            item_type (Optional[int]): The community item type of the game profile.

        Returns:
            Optional[dict[str, str]]: The theme colors, or None if the preview page has none.
        """
        if time.monotonic() < self.previews_paused_until:
            raise requests.HTTPError("Profile previews are paused after being rate limited by Steam")

        for attempt in range(PROFILE_PREVIEW_ATTEMPTS):
            wait = PROFILE_PREVIEW_INTERVAL_SECONDS - (time.monotonic() - self.last_preview_request)

            if wait > 0:
                time.sleep(wait)

            self.last_preview_request = time.monotonic()
            response = self.session.get(
                settings.PROFILE_PREVIEW_URL,
                params={"previewprofile": 1, "appid": app_id, "itemtype": item_type},
                timeout=30,
            )

            if response.status_code != 429:
                break

            if attempt == PROFILE_PREVIEW_ATTEMPTS - 1:
                self.previews_paused_until = time.monotonic() + PROFILE_PREVIEW_COOLDOWN_SECONDS
                break

            retry_after = response.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else PROFILE_PREVIEW_BACKOFF_SECONDS * (attempt + 1)
            logger.info("Steam is rate limiting profile previews, retrying in %s seconds", delay)
            time.sleep(delay)

        response.raise_for_status()
        match = re.search(r"body\.GameProfileTheme\s*\{([^}]*)\}", response.text)

        return self._parse_theme_colors(match.group(1)) if match else None

    def get_game_profile(self, definition: dict[str, Any]) -> dict[str, Any] | None:
        """
        Retrieves the theme and the bundled items a game profile applies to a Steam profile.

        Args:
            definition (dict[str, Any]): The raw item definition of the game profile.

        Returns:
            Optional[dict[str, Any]]: The theme and bundled item assets, or None if they could not be fetched.
        """
        bundle_ids = definition.get("bundle_defids") or []
        theme_id = definition.get("community_item_data", {}).get("profile_theme_id")
        parts: list[dict[str, Any]] = []

        try:
            if bundle_ids:
                response = self.session.get(
                    API_BASE_URL,
                    params={
                        "include_direct_purchase_disabled": "true",
                        **{f"definitionids[{index}]": bundle_id for index, bundle_id in enumerate(bundle_ids)},
                    },
                    timeout=30,
                )
                response.raise_for_status()
                parts = response.json().get("response", {}).get("definitions", [])

            if theme_id == "GameProfile":
                colors = self._get_game_profile_theme(
                    definition.get("appid"), definition.get("community_item_type"))
            else:
                colors = self._get_named_theme(theme_id) if theme_id else None
        except (requests.RequestException, ValueError) as e:
            logger.warning("Could not fetch game profile %s: %s", definition.get("defid"), e)
            return None

        profile: dict[str, Any] = {
            "theme": {"name": theme_id, "colors": colors},
            **{name: None for name in BUNDLE_PARTS.values()},
        }

        for part in parts:
            name = BUNDLE_PARTS.get(part.get("community_item_class"))

            if name:
                data = part.get("community_item_data", {})
                profile[name] = {
                    **self._map_assets(part.get("appid"), data),
                    "animated": bool(data.get("animated")),
                }

        return profile
