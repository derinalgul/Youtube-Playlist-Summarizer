"""Tests for the API endpoints."""

import os
import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file for OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

from httpx import AsyncClient, ASGITransport

from backend.api.app import create_app
from backend.api.schemas import JobStatus


# Check if OpenAI API key is available
HAS_OPENAI_KEY = os.environ.get("OPENAI_API_KEY") is not None
skip_without_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY,
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    # Set environment variable for cache directory
    os.environ["YKB_CACHE_DIR"] = temp_dir
    yield Path(temp_dir)
    # Cleanup
    if "YKB_CACHE_DIR" in os.environ:
        del os.environ["YKB_CACHE_DIR"]
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def client(temp_cache_dir):
    """Create a test client with app lifespan."""
    from contextlib import asynccontextmanager
    from backend.api.app import lifespan, get_services
    from backend.services.video_processor import VideoProcessor
    from backend.services.transcription import TranscriptionService
    from backend.services.embeddings import EmbeddingsService
    from backend.services.query_engine import QueryEngine
    from backend.api.jobs import init_job_manager

    app = create_app()

    # Manually initialize services for testing (since lifespan won't auto-trigger)
    video_processor = VideoProcessor()
    transcription_service = TranscriptionService(video_processor=video_processor)
    embeddings_service = EmbeddingsService(
        persist_directory=temp_cache_dir / "chroma",
        collection_name="test_collection",
    )
    query_engine = QueryEngine(embeddings_service=embeddings_service)
    job_manager = init_job_manager(
        video_processor=video_processor,
        transcription_service=transcription_service,
        embeddings_service=embeddings_service,
    )

    # Store services globally
    services = get_services()
    services["video_processor"] = video_processor
    services["transcription"] = transcription_service
    services["embeddings"] = embeddings_service
    services["query_engine"] = query_engine
    services["job_manager"] = job_manager

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    # Cleanup
    embeddings_service.close()


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test that health endpoint returns ok status."""
        response = await client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "services" in data


@skip_without_openai
class TestVideoSubmission:
    """Tests for video submission endpoint."""

    @pytest.mark.asyncio
    async def test_submit_video_invalid_url(self, client):
        """Test that invalid URLs are rejected."""
        response = await client.post(
            "/api/videos",
            json={"url": "not-a-valid-url"},
        )
        assert response.status_code == 400
        assert "Invalid YouTube URL" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_submit_video_valid_url(self, client):
        """Test submitting a valid video URL."""
        response = await client.post(
            "/api/videos",
            json={"url": "https://www.youtube.com/watch?v=ynvUiKy1BmA"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "job_id" in data
        assert data["status"] in [s.value for s in JobStatus]
        assert "created_at" in data


@skip_without_openai
class TestJobStatus:
    """Tests for job status endpoint."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, client):
        """Test that nonexistent job returns 404."""
        response = await client.get("/api/jobs/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_jobs(self, client):
        """Test listing all jobs."""
        response = await client.get("/api/jobs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@skip_without_openai
class TestQueryEndpoint:
    """Tests for the query endpoint."""

    @pytest.mark.asyncio
    async def test_query_empty_collection(self, client):
        """Test querying when no videos are indexed."""
        response = await client.post(
            "/api/query",
            json={"question": "What is Python?"},
        )
        # Should return 404 because nothing is indexed
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_query_validation(self, client):
        """Test query validation."""
        # Empty question should fail
        response = await client.post(
            "/api/query",
            json={"question": ""},
        )
        assert response.status_code == 422  # Validation error


@skip_without_openai
class TestStatsEndpoint:
    """Tests for the stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats(self, client):
        """Test getting collection statistics."""
        response = await client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_chunks" in data
        assert "total_videos" in data
        assert "collection_name" in data


class TestClearHistory:
    """Tests for clearing conversation history."""

    @pytest.mark.asyncio
    @skip_without_openai
    async def test_clear_history(self, client):
        """Test clearing conversation history."""
        response = await client.delete("/api/query/history")
        assert response.status_code == 200
        assert "message" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
