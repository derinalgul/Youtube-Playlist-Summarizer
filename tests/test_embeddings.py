"""Tests for the EmbeddingsService module."""

import os
import pytest
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file for OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

from backend.services.embeddings import (
    EmbeddingsService,
    TextChunk,
    SearchResult,
    chunk_transcript,
)
from backend.models.video import CaptionSegment, CaptionType, VideoCaption


# Test video ID - same as other tests
TEST_VIDEO_ID = "ynvUiKy1BmA"

# Check if OpenAI API key is available
HAS_OPENAI_KEY = os.environ.get("OPENAI_API_KEY") is not None
skip_without_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY,
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def sample_caption():
    """Create a sample VideoCaption for testing."""
    segments = [
        CaptionSegment(text="Hello and welcome to this video.", start=0.0, duration=3.0),
        CaptionSegment(text="Today we will learn about Python.", start=3.0, duration=3.5),
        CaptionSegment(text="Python is a programming language.", start=6.5, duration=3.0),
        CaptionSegment(text="It is very popular for data science.", start=9.5, duration=3.5),
        CaptionSegment(text="Let's start with the basics.", start=13.0, duration=2.5),
        CaptionSegment(text="Variables store data in memory.", start=15.5, duration=3.0),
        CaptionSegment(text="You can use integers, strings, and floats.", start=18.5, duration=4.0),
        CaptionSegment(text="Functions help organize your code.", start=22.5, duration=3.0),
        CaptionSegment(text="They can take parameters and return values.", start=25.5, duration=4.0),
        CaptionSegment(text="That's all for today, thanks for watching!", start=29.5, duration=3.5),
    ]

    return VideoCaption(
        video_id=TEST_VIDEO_ID,
        language="en",
        caption_type=CaptionType.AUTO_GENERATED,
        segments=segments,
        full_text=" ".join(s.text for s in segments),
    )


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestChunkTranscript:
    """Tests for the chunk_transcript function."""

    def test_chunk_basic(self, sample_caption):
        """Test basic chunking functionality."""
        chunks = chunk_transcript(sample_caption, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.video_id == TEST_VIDEO_ID for c in chunks)

    def test_chunk_preserves_timestamps(self, sample_caption):
        """Test that chunks preserve timestamp information."""
        chunks = chunk_transcript(sample_caption, chunk_size=100, chunk_overlap=20)

        for chunk in chunks:
            assert chunk.start_time >= 0
            assert chunk.end_time > chunk.start_time
            assert chunk.chunk_index >= 0

    def test_chunk_indices_sequential(self, sample_caption):
        """Test that chunk indices are sequential."""
        chunks = chunk_transcript(sample_caption, chunk_size=100, chunk_overlap=20)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_caption(self):
        """Test chunking an empty caption."""
        empty_caption = VideoCaption(
            video_id="test",
            language="en",
            caption_type=CaptionType.AUTO_GENERATED,
            segments=[],
            full_text="",
        )

        chunks = chunk_transcript(empty_caption)
        assert chunks == []

    def test_small_caption_single_chunk(self):
        """Test that a small caption results in a single chunk."""
        small_caption = VideoCaption(
            video_id="test",
            language="en",
            caption_type=CaptionType.AUTO_GENERATED,
            segments=[
                CaptionSegment(text="Short text.", start=0.0, duration=2.0),
            ],
            full_text="Short text.",
        )

        chunks = chunk_transcript(small_caption, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."


@skip_without_openai
class TestEmbeddingsServiceInit:
    """Tests for EmbeddingsService initialization."""

    def test_init_creates_collection(self, temp_chroma_dir):
        """Test that initialization creates a ChromaDB collection."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_collection",
        ) as service:
            stats = service.get_collection_stats()
            assert stats["collection_name"] == "test_collection"
            assert stats["total_chunks"] == 0


@skip_without_openai
class TestEmbeddingsServiceIndexing:
    """Tests for video indexing functionality."""

    def test_index_video(self, sample_caption, temp_chroma_dir):
        """Test indexing a video transcript."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_index",
        ) as service:
            num_chunks = service.index_video(
                sample_caption,
                video_title="Test Video",
                chunk_size=100,
            )

            assert num_chunks > 0
            assert service.is_video_indexed(TEST_VIDEO_ID)

    def test_is_video_indexed_false(self, temp_chroma_dir):
        """Test is_video_indexed returns False for non-indexed video."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_not_indexed",
        ) as service:
            assert service.is_video_indexed("nonexistent_video") is False

    def test_delete_video(self, sample_caption, temp_chroma_dir):
        """Test deleting a video from the index."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_delete",
        ) as service:
            # Index first
            service.index_video(sample_caption, chunk_size=100)
            assert service.is_video_indexed(TEST_VIDEO_ID)

            # Delete
            deleted = service.delete_video(TEST_VIDEO_ID)
            assert deleted > 0
            assert service.is_video_indexed(TEST_VIDEO_ID) is False


@skip_without_openai
class TestEmbeddingsServiceSearch:
    """Tests for semantic search functionality."""

    def test_search_returns_results(self, sample_caption, temp_chroma_dir):
        """Test that search returns relevant results."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_search",
        ) as service:
            # Index the video
            service.index_video(
                sample_caption,
                video_title="Python Tutorial",
                chunk_size=100,
            )

            # Search for something in the transcript
            results = service.search("What is Python?", top_k=3)

            assert len(results) > 0
            assert all(isinstance(r, SearchResult) for r in results)
            assert all(r.score > 0 for r in results)

    def test_search_result_has_metadata(self, sample_caption, temp_chroma_dir):
        """Test that search results include proper metadata."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_search_meta",
        ) as service:
            service.index_video(
                sample_caption,
                video_title="Test Video",
                chunk_size=100,
            )

            results = service.search("Python programming", top_k=1)

            assert len(results) > 0
            result = results[0]
            assert result.chunk.video_id == TEST_VIDEO_ID
            assert result.chunk.start_time >= 0
            assert result.chunk.end_time > 0
            assert len(result.chunk.text) > 0

    def test_search_with_video_filter(self, sample_caption, temp_chroma_dir):
        """Test filtering search results by video ID."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_filter",
        ) as service:
            service.index_video(sample_caption, chunk_size=100)

            # Search with matching filter
            results = service.search(
                "Python",
                top_k=5,
                video_ids=[TEST_VIDEO_ID],
            )
            assert len(results) > 0

            # Search with non-matching filter
            results = service.search(
                "Python",
                top_k=5,
                video_ids=["nonexistent_video"],
            )
            assert len(results) == 0


@skip_without_openai
class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self, temp_chroma_dir):
        """Test using EmbeddingsService as context manager."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_context",
        ) as service:
            assert service is not None
            stats = service.get_collection_stats()
            assert "total_chunks" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
