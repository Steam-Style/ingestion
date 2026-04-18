from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "http://steam-style-qdrant:6333"
    COLLECTION_NAME: str = "Steam-Style-Items"
    MODEL_NAME: str = "google/siglip2-base-patch16-224"
    IMAGE_EMBEDDING_DIM: int = 768
    DEVICE: str = "cpu"

    COLOR_HUE_BINS: int = 24
    COLOR_SAT_BINS: int = 3
    COLOR_VAL_BINS: int = 3
    COLOR_SIGMA_H: float = 0.05
    COLOR_SIGMA_S: float = 0.25
    COLOR_SIGMA_V: float = 0.25
    COLOR_POWER: float = 0.5


settings = Settings()
