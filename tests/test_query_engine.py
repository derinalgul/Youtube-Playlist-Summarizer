"""Tests for the QueryEngine module."""

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

from backend.services.query_engine import (
    Citation,
    ConversationHistory,
    Message,
    NoResultsError,
    QueryEngine,
    QueryResponse,
)
from backend.services.embeddings import EmbeddingsService
from backend.models.video import CaptionSegment, CaptionType, VideoCaption


# Test video ID
TEST_VIDEO_ID = "test_video_123"

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
        CaptionSegment(text="Welcome to this Python tutorial.", start=0.0, duration=3.0),
        CaptionSegment(text="Today we will learn about variables.", start=3.0, duration=3.5),
        CaptionSegment(text="A variable is a container for storing data.", start=6.5, duration=4.0),
        CaptionSegment(text="You can create a variable by using the equals sign.", start=10.5, duration=4.0),
        CaptionSegment(text="For example, x equals 5 creates a variable named x.", start=14.5, duration=4.5),
        CaptionSegment(text="Python has different data types like integers and strings.", start=19.0, duration=4.0),
        CaptionSegment(text="Strings are text enclosed in quotes.", start=23.0, duration=3.5),
        CaptionSegment(text="You can also use functions to organize code.", start=26.5, duration=4.0),
        CaptionSegment(text="Functions are defined using the def keyword.", start=30.5, duration=3.5),
        CaptionSegment(text="Thanks for watching this tutorial!", start=34.0, duration=3.0),
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


class TestCitation:
    """Tests for Citation class."""

    def test_format_timestamp_minutes(self):
        """Test timestamp formatting for times under an hour."""
        citation = Citation(
            video_id="abc123",
            video_title="Test Video",
            start_time=125.5,  # 2:05
            end_time=130.0,
            text_snippet="Some text",
        )
        assert citation.format_timestamp() == "2:05"

    def test_format_timestamp_hours(self):
        """Test timestamp formatting for times over an hour."""
        citation = Citation(
            video_id="abc123",
            video_title="Test Video",
            start_time=3725.0,  # 1:02:05
            end_time=3730.0,
            text_snippet="Some text",
        )
        assert citation.format_timestamp() == "1:02:05"

    def test_format_link(self):
        """Test YouTube link formatting."""
        citation = Citation(
            video_id="abc123",
            video_title="Test Video",
            start_time=125.5,
            end_time=130.0,
            text_snippet="Some text",
        )
        assert citation.format_link() == "https://youtu.be/abc123?t=125"

    def test_format_reference(self):
        """Test reference formatting."""
        citation = Citation(
            video_id="abc123",
            video_title="Python Basics",
            start_time=65.0,  # 1:05
            end_time=70.0,
            text_snippet="Some text",
        )
        assert citation.format_reference() == "[Python Basics @ 1:05]"


class TestConversationHistory:
    """Tests for ConversationHistory class."""

    def test_add_messages(self):
        """Test adding messages to history."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi there!")

        assert len(history.messages) == 2
        assert history.messages[0].role == "user"
        assert history.messages[1].role == "assistant"

    def test_trim_history(self):
        """Test that history is trimmed to max_messages."""
        history = ConversationHistory(max_messages=3)

        for i in range(5):
            history.add_user_message(f"Message {i}")

        assert len(history.messages) == 3
        assert history.messages[0].content == "Message 2"

    def test_get_messages_for_api(self):
        """Test conversion to API format."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi!")

        api_messages = history.get_messages_for_api()

        assert api_messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_clear_history(self):
        """Test clearing history."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.clear()

        assert len(history.messages) == 0


@skip_without_openai
class TestQueryEngine:
    """Tests for QueryEngine class."""

    def test_query_returns_response(self, sample_caption, temp_chroma_dir):
        """Test that query returns a valid response."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_query",
        ) as embeddings:
            # Index the sample video
            embeddings.index_video(
                sample_caption,
                video_title="Python Tutorial",
                chunk_size=100,
            )

            # Create query engine
            engine = QueryEngine(embeddings)

            # Query
            response = engine.query("What is a variable?")

            assert isinstance(response, QueryResponse)
            assert len(response.answer) > 0
            assert len(response.citations) > 0
            assert response.model_used == "gpt-4o-mini"

    def test_query_response_has_citations(self, sample_caption, temp_chroma_dir):
        """Test that response includes proper citations."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_citations",
        ) as embeddings:
            embeddings.index_video(
                sample_caption,
                video_title="Python Tutorial",
                chunk_size=100,
            )

            engine = QueryEngine(embeddings)
            response = engine.query("How do I create a variable?")

            assert len(response.citations) > 0
            citation = response.citations[0]
            assert citation.video_id == TEST_VIDEO_ID
            assert citation.video_title == "Python Tutorial"
            assert citation.start_time >= 0

    def test_conversation_history(self, sample_caption, temp_chroma_dir):
        """Test that conversation history is maintained."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_history",
        ) as embeddings:
            embeddings.index_video(
                sample_caption,
                video_title="Python Tutorial",
                chunk_size=100,
            )

            engine = QueryEngine(embeddings)

            # First query
            engine.query("What is a variable?")

            # Check history
            history = engine.get_history()
            assert len(history) == 2  # user + assistant

            # Second query
            engine.query("How do I create one?")

            history = engine.get_history()
            assert len(history) == 4  # 2 user + 2 assistant

    def test_clear_history(self, sample_caption, temp_chroma_dir):
        """Test clearing conversation history."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_clear",
        ) as embeddings:
            embeddings.index_video(
                sample_caption,
                video_title="Python Tutorial",
                chunk_size=100,
            )

            engine = QueryEngine(embeddings)
            engine.query("What is Python?")

            engine.clear_history()
            assert len(engine.get_history()) == 0

    def test_no_results_error(self, temp_chroma_dir):
        """Test that NoResultsError is raised when no content is indexed."""
        with EmbeddingsService(
            persist_directory=temp_chroma_dir,
            collection_name="test_empty",
        ) as embeddings:
            engine = QueryEngine(embeddings)

            with pytest.raises(NoResultsError):
                engine.query("What is Python?")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
