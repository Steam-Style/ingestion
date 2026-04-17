"""
Color embedding module using soft-assignment against a fixed HSV palette.
"""
from typing import Union, List

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.color import rgb2hsv


class ColorEmbedder:
    def __init__(
        self,
        hue_bins: int = 24,
        sat_bins: int = 3,
        val_bins: int = 3,
        sigma_h: float = 0.05,
        sigma_s: float = 0.25,
        sigma_v: float = 0.25,
        power: float = 0.5,
    ) -> None:
        self.sigma_h = sigma_h
        self.sigma_s = sigma_s
        self.sigma_v = sigma_v
        self.power = power
        self._palette_hsv = self._build_palette(hue_bins, sat_bins, val_bins)

    def _build_palette(
        self, h_bins: int, s_bins: int, v_bins: int
    ) -> NDArray[np.float64]:
        h_vals = np.linspace(0, 1, h_bins, endpoint=False)
        s_vals = np.linspace(1 / s_bins, 1.0, s_bins)
        v_vals = np.linspace(1 / v_bins, 1.0, v_bins)
        h, s, v = np.meshgrid(h_vals, s_vals, v_vals, indexing="ij")
        return np.stack([h.ravel(), s.ravel(), v.ravel()], axis=1)

    @property
    def embedding_dimension(self) -> int:
        return len(self._palette_hsv)

    def _soft_histogram(self, hsv_pixels: NDArray[np.float64]) -> NDArray[np.float64]:
        dh = np.abs(
            hsv_pixels[:, np.newaxis, 0] - self._palette_hsv[np.newaxis, :, 0]
        )
        dh = np.minimum(dh, 1.0 - dh)

        ds = hsv_pixels[:, np.newaxis, 1] - self._palette_hsv[np.newaxis, :, 1]
        dv = hsv_pixels[:, np.newaxis, 2] - self._palette_hsv[np.newaxis, :, 2]

        dist_sq = (
            (dh / self.sigma_h) ** 2
            + (ds / self.sigma_s) ** 2
            + (dv / self.sigma_v) ** 2
        )
        weights = np.exp(-dist_sq / 2.0)
        vector = weights.sum(axis=0)

        if self.power != 1.0:
            vector = np.power(vector, self.power)

        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            vector /= norm
        return vector

    def _extract_chromatic_hsv(
        self, image: Image.Image, alpha_mask: NDArray[np.uint8] | None
    ) -> NDArray[np.float64]:
        rgb_float = np.array(image.convert("RGB")).astype(np.float64) / 255.0
        hsv_full = rgb2hsv(rgb_float)

        if alpha_mask is not None:
            opaque = alpha_mask > 128
            if opaque.any():
                hsv_flat = hsv_full[opaque]
            else:
                hsv_flat = hsv_full.reshape(-1, 3)
        else:
            hsv_flat = hsv_full.reshape(-1, 3)

        chromatic = (hsv_flat[:, 1] >= 0.12) & (hsv_flat[:, 2] >= 0.08)
        if chromatic.sum() >= 10:
            return hsv_flat[chromatic]
        return hsv_flat

    def image_to_embedding(
        self, image_source: Union[str, Image.Image]
    ) -> NDArray[np.float64]:
        if isinstance(image_source, str):
            image = Image.open(image_source)
        else:
            image = image_source

        image = image.resize((128, 128))

        alpha_mask = None
        if image.mode in ("RGBA", "LA"):
            alpha_mask = np.array(image.split()[-1])
        elif image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
            alpha_mask = np.array(image.split()[-1])

        hsv_pixels = self._extract_chromatic_hsv(image, alpha_mask)
        return self._soft_histogram(hsv_pixels)

    def query_to_embedding(self, hex_colors: List[str]) -> list[float]:
        rgb_colors = []

        for hex_str in hex_colors:
            clean = hex_str.lstrip("#")
            if len(clean) == 3:
                clean = "".join(c * 2 for c in clean)
            if len(clean) != 6:
                raise ValueError(f"Invalid hex color: {hex_str}")
            rgb_colors.append([
                int(clean[0:2], 16),
                int(clean[2:4], 16),
                int(clean[4:6], 16),
            ])

        rgb_arr = np.array(rgb_colors, dtype=np.float64) / 255.0
        hsv_query = rgb2hsv(rgb_arr.reshape(-1, 1, 3)).reshape(-1, 3)
        return self._soft_histogram(hsv_query).tolist()
