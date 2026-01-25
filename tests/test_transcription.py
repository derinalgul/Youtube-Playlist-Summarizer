"""Tests for the TranscriptionService module."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.transcription import (
    TranscriptionService,
)
from backend.models.video import CaptionType, VideoCaption


# Test video - short Python tutorial with auto-generated captions
TEST_VIDEO_ID = "ynvUiKy1BmA"


@pytest.fixture
def service():
    """Create a TranscriptionService instance for testing."""
    svc = TranscriptionService()
    yield svc
    svc.close()


class TestTranscriptionServiceInit:
    """Tests for TranscriptionService initialization."""

    def test_default_init(self):
        """Test default initialization."""
        with TranscriptionService() as service:
            assert service is not None


class TestGetTranscript:
    """Tests for transcript retrieval."""

    def test_get_transcript(self, service):
        """Test getting transcript for a video with captions."""
        transcript = service.get_transcript(TEST_VIDEO_ID)

        assert isinstance(transcript, VideoCaption)
        assert transcript.video_id == TEST_VIDEO_ID
        assert transcript.caption_type in [CaptionType.MANUAL, CaptionType.AUTO_GENERATED]
        assert len(transcript.segments) > 0
        assert len(transcript.full_text) > 0

    def test_transcript_has_timestamps(self, service):
        """Test that transcript segments have proper timestamps."""
        transcript = service.get_transcript(TEST_VIDEO_ID)

        for segment in transcript.segments:
            assert segment.start >= 0
            assert segment.duration > 0
            assert len(segment.text) > 0

    def test_transcript_caching(self, service):
        """Test that transcripts are cached."""
        # First fetch
        transcript1 = service.get_transcript(TEST_VIDEO_ID)

        # Second fetch should use cache
        transcript2 = service.get_transcript(TEST_VIDEO_ID)

        assert transcript1.video_id == transcript2.video_id
        assert transcript1.full_text == transcript2.full_text


class TestCaptionAvailability:
    """Tests for caption availability checking."""

    def test_has_captions(self, service):
        """Test checking if video has captions."""
        has_caps = service.has_captions(TEST_VIDEO_ID)
        assert has_caps is True

    def test_list_available_languages(self, service):
        """Test listing available caption languages."""
        languages = service.list_available_languages(TEST_VIDEO_ID)

        assert "manual" in languages
        assert "auto_generated" in languages
        # This video has auto-generated English
        assert len(languages["auto_generated"]) > 0 or len(languages["manual"]) > 0


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self):
        """Test using TranscriptionService as context manager."""
        with TranscriptionService() as service:
            transcript = service.get_transcript(TEST_VIDEO_ID)
            assert transcript is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
