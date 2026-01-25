"""Transcription service for YouTube Knowledge Base.

Provides a unified interface for getting transcripts from YouTube videos
using YouTube's built-in captions (manual or auto-generated).
"""

import logging
from typing import Optional

from backend.models.video import VideoCaption
from backend.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Base exception for transcription errors."""

    def __init__(self, message: str, video_id: Optional[str] = None):
        self.message = message
        self.video_id = video_id
        super().__init__(self.message)


class NoCaptionsAvailableError(TranscriptionError):
    """Raised when no captions are available for a video."""
    pass


class TranscriptionService:
    """
    Service for getting video transcripts.

    Uses YouTube's built-in captions (manual or auto-generated).
    Most YouTube videos (~95%) have auto-generated captions available.

    Usage:
        service = TranscriptionService()
        transcript = service.get_transcript("video_id")
    """

    def __init__(self, video_processor: Optional[VideoProcessor] = None):
        """
        Initialize the TranscriptionService.

        Args:
            video_processor: VideoProcessor instance (creates one if not provided)
        """
        self._video_processor = video_processor or VideoProcessor()
        self._owns_video_processor = video_processor is None

    def get_transcript(
        self,
        video_id: str,
        use_cache: bool = True,
    ) -> VideoCaption:
        """
        Get transcript for a video using YouTube captions.

        Priority:
        1. Manual YouTube captions
        2. Auto-generated YouTube captions

        Args:
            video_id: YouTube video ID
            use_cache: Use cached data where available

        Returns:
            VideoCaption with transcript data

        Raises:
            NoCaptionsAvailableError: If no captions are available
            TranscriptionError: If transcription fails
        """
        try:
            caption = self._video_processor.get_captions(video_id, use_cache=use_cache)

            if caption is None:
                raise NoCaptionsAvailableError(
                    f"No captions available for video {video_id}",
                    video_id=video_id,
                )

            logger.info(
                f"Using YouTube {caption.caption_type.value} captions for {video_id}"
            )
            return caption

        except NoCaptionsAvailableError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Failed to get transcript: {e}",
                video_id=video_id,
            ) from e

    def has_captions(self, video_id: str) -> bool:
        """
        Check if a video has captions available.

        Args:
            video_id: YouTube video ID

        Returns:
            True if captions are available, False otherwise
        """
        manual_langs, auto_langs = self._video_processor.list_available_captions(
            video_id
        )
        return len(manual_langs) > 0 or len(auto_langs) > 0

    def list_available_languages(self, video_id: str) -> dict:
        """
        List available caption languages for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dict with 'manual' and 'auto_generated' language lists
        """
        manual_langs, auto_langs = self._video_processor.list_available_captions(
            video_id
        )
        return {
            "manual": manual_langs,
            "auto_generated": auto_langs,
        }

    def close(self):
        """Clean up resources."""
        if self._owns_video_processor:
            self._video_processor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
