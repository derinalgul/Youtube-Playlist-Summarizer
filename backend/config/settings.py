"""Configuration settings for the YouTube Knowledge Base."""

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Base directories
    base_dir: Path = Path(__file__).parent.parent.parent
    cache_dir: Path = Path(".cache")
    audio_output_dir: Path = Path(".cache/audio")

    # Video processor settings
    preferred_languages: List[str] = ["en", "en-US", "en-GB"]

    # Cache TTL (in seconds)
    metadata_cache_ttl: int = 7 * 24 * 60 * 60  # 7 days
    captions_cache_ttl: int = 30 * 24 * 60 * 60  # 30 days
    playlist_cache_ttl: int = 1 * 24 * 60 * 60  # 1 day

    # yt-dlp settings
    ytdlp_quiet: bool = True
    audio_format: str = "mp3"
    audio_quality: str = "192"

    class Config:
        env_prefix = "YKB_"  # YouTube Knowledge Base
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars like OPENAI_API_KEY


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (lazy initialization)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Backwards compatibility
settings = get_settings()
