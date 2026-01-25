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
from backend.services.transcription import (
    NoCaptionsAvailableError,
    TranscriptionError,
    TranscriptionService,
)

__all__ = [
    # Video Processor
    "AudioExtractionError",
    "CaptionExtractionError",
    "MetadataExtractionError",
    "RateLimitError",
    "URLParsingError",
    "VideoProcessor",
    "VideoProcessorError",
    "VideoUnavailableError",
    # Transcription Service
    "NoCaptionsAvailableError",
    "TranscriptionError",
    "TranscriptionService",
]