"""Services module for the YouTube Knowledge Base."""

from backend.services.video_processor import (
    AudioExtractionError,
    CaptionExtractionError,
    MetadataExtractionError,
    RateLimitError,
    URLParsingError,
    VideoProcessor,
    VideoProcessorError,
    VideoUnavailableError,
)

__all__ = [
    "AudioExtractionError",
    "CaptionExtractionError",
    "MetadataExtractionError",
    "RateLimitError",
    "URLParsingError",
    "VideoProcessor",
    "VideoProcessorError",
    "VideoUnavailableError",
]