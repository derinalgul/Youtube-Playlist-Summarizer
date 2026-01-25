"""Tests for the VideoProcessor module."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.video_processor import VideoProcessor
from backend.models.video import (
    VideoSource,
    CaptionType,
    URLParseResult,
    VideoMetadata,
    VideoCaption,
    ProcessedVideo,
)


# Test video - short Python tutorial
TEST_VIDEO_ID = "ynvUiKy1BmA"
TEST_VIDEO_URL = f"https://youtu.be/{TEST_VIDEO_ID}"
TEST_VIDEO_TITLE = "Output Hello World in Python / How to Tutorial"


@pytest.fixture
def processor():
    """Create a VideoProcessor instance for testing."""
    proc = VideoProcessor()
    yield proc
    proc.close()


class TestURLParsing:
    """Tests for URL parsing functionality."""

    def test_parse_standard_url(self, processor):
        """Test parsing standard youtube.com/watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = processor.parse_url(url)

        assert result.is_valid
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.source_type == VideoSource.SINGLE_VIDEO

    def test_parse_short_url(self, processor):
        """Test parsing youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = processor.parse_url(url)

        assert result.is_valid
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.source_type == VideoSource.SINGLE_VIDEO

    def test_parse_url_with_timestamp(self, processor):
        """Test parsing URL with timestamp parameter."""
        url = "https://youtu.be/ynvUiKy1BmA?si=_pnnZvUDDDcHaxDT"
        result = processor.parse_url(url)

        assert result.is_valid
        assert result.video_id == "ynvUiKy1BmA"

    def test_parse_playlist_url(self, processor):
        """Test parsing playlist URL."""
        url = "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        result = processor.parse_url(url)

        assert result.is_valid
        assert result.playlist_id == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        assert result.source_type == VideoSource.PLAYLIST

    def test_parse_video_with_playlist_context(self, processor):
        """Test parsing video URL that includes playlist context."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        result = processor.parse_url(url)

        assert result.is_valid
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.playlist_id == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        # Single video takes precedence
        assert result.source_type == VideoSource.SINGLE_VIDEO

    def test_parse_embed_url(self, processor):
        """Test parsing embed URL format."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        result = processor.parse_url(url)

        assert result.is_valid
        assert result.video_id == "dQw4w9WgXcQ"

    def test_parse_invalid_url(self, processor):
        """Test parsing invalid URL."""
        url = "https://example.com/not-a-youtube-url"
        result = processor.parse_url(url)

        assert not result.is_valid
        assert result.error_message is not None


class TestMetadataExtraction:
    """Tests for video metadata extraction."""

    def test_get_video_metadata(self, processor):
        """Test fetching metadata for a real video."""
        metadata = processor.get_video_metadata(TEST_VIDEO_ID)

        assert isinstance(metadata, VideoMetadata)
        assert metadata.video_id == TEST_VIDEO_ID
        assert metadata.title == TEST_VIDEO_TITLE
        assert metadata.channel_name == "RubenOrtega"
        assert metadata.duration_seconds > 0

    def test_metadata_caching(self, processor):
        """Test that metadata is cached on second fetch."""
        # First fetch
        metadata1 = processor.get_video_metadata(TEST_VIDEO_ID)

        # Second fetch should use cache
        metadata2 = processor.get_video_metadata(TEST_VIDEO_ID)

        assert metadata1.video_id == metadata2.video_id
        assert metadata1.title == metadata2.title

    def test_metadata_youtube_url_property(self, processor):
        """Test the youtube_url property."""
        metadata = processor.get_video_metadata(TEST_VIDEO_ID)

        assert TEST_VIDEO_ID in metadata.youtube_url
        assert "youtube.com/watch" in metadata.youtube_url


class TestCaptionExtraction:
    """Tests for caption extraction."""

    def test_get_captions(self, processor):
        """Test fetching captions for a video with auto-generated captions."""
        caption = processor.get_captions(TEST_VIDEO_ID)

        assert caption is not None
        assert isinstance(caption, VideoCaption)
        assert caption.video_id == TEST_VIDEO_ID
        assert caption.language == "en"
        assert caption.caption_type == CaptionType.AUTO_GENERATED
        assert len(caption.segments) > 0
        assert len(caption.full_text) > 0

    def test_caption_segments_have_timestamps(self, processor):
        """Test that caption segments have proper timestamps."""
        caption = processor.get_captions(TEST_VIDEO_ID)

        for segment in caption.segments:
            assert segment.start >= 0
            assert segment.duration > 0
            assert len(segment.text) > 0

    def test_list_available_captions(self, processor):
        """Test listing available caption languages."""
        manual_langs, auto_langs = processor.list_available_captions(TEST_VIDEO_ID)

        # This video has auto-generated English captions
        assert "en" in auto_langs or len(auto_langs) > 0


class TestCacheManagement:
    """Tests for cache management."""

    def test_is_cached(self, processor):
        """Test checking if data is cached."""
        # Ensure metadata is fetched and cached
        processor.get_video_metadata(TEST_VIDEO_ID)

        assert processor.is_cached(TEST_VIDEO_ID, "metadata")

    def test_clear_specific_cache(self, processor):
        """Test clearing cache for a specific video."""
        # Ensure metadata is cached
        processor.get_video_metadata(TEST_VIDEO_ID)
        assert processor.is_cached(TEST_VIDEO_ID, "metadata")

        # Clear the cache
        cleared = processor.clear_cache(TEST_VIDEO_ID, "metadata")

        assert cleared == 1
        assert not processor.is_cached(TEST_VIDEO_ID, "metadata")

    def test_get_cache_stats(self, processor):
        """Test getting cache statistics."""
        # Ensure something is cached
        processor.get_video_metadata(TEST_VIDEO_ID)

        stats = processor.get_cache_stats()

        assert "size_bytes" in stats
        assert "entry_count" in stats
        assert "directory" in stats
        assert stats["entry_count"] >= 1


class TestHighLevelProcessing:
    """Tests for high-level processing methods."""

    def test_process_video(self, processor):
        """Test the full video processing pipeline."""
        result = processor.process_video(TEST_VIDEO_ID)

        assert isinstance(result, ProcessedVideo)
        assert result.metadata.video_id == TEST_VIDEO_ID
        assert result.processing_status == "completed"
        assert result.caption is not None

    def test_process_url(self, processor):
        """Test processing a URL."""
        results = processor.process_url(TEST_VIDEO_URL)

        assert len(results) == 1
        assert results[0].metadata.video_id == TEST_VIDEO_ID
        assert results[0].processing_status == "completed"


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self):
        """Test using VideoProcessor as context manager."""
        with VideoProcessor() as processor:
            result = processor.parse_url(TEST_VIDEO_URL)
            assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
