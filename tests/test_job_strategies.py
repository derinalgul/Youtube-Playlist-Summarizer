"""Tests for job processing strategies (parallel, bounded, sequential).

Uses a ConcurrencyTracker that simulates rate limiting when too many
videos are processed concurrently, which naturally exercises each strategy:
- Parallel (all at once): exceeds the limit → triggers fallback
- Bounded (semaphore of 3): may or may not exceed depending on limit
- Sequential (one at a time): never exceeds
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.jobs import JobManager, _is_rate_limit_error
from backend.api.schemas import JobStatus, VideoInfo
from backend.services.video_processor import RateLimitError


class ConcurrencyTracker:
    """Simulates rate limiting when concurrent calls exceed a threshold.

    When more than ``max_concurrent`` calls are active simultaneously,
    new calls raise a RateLimitError — mirroring how real APIs behave
    when flooded with too many requests at once.
    """

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self._active = 0

    async def mock_process_video(self, job, video_id):
        """Drop-in replacement for JobManager._process_video."""
        # Create VideoInfo like the real method does after fetching metadata
        job.videos[video_id] = VideoInfo(
            video_id=video_id,
            title=f"Video {video_id}",
            channel="Test Channel",
            duration_seconds=100,
            indexed=False,
            chunks_count=0,
        )

        self._active += 1
        if self._active > self.max_concurrent:
            self._active -= 1
            raise RateLimitError("Too many concurrent requests", retry_after=60)

        # Simulate I/O work — yields control so other tasks can start
        await asyncio.sleep(0.01)

        job.videos[video_id].indexed = True
        job.videos[video_id].chunks_count = 5
        self._active -= 1


def _create_job_manager() -> JobManager:
    """Create a JobManager with mock services (services aren't called directly)."""
    return JobManager(
        video_processor=MagicMock(),
        transcription_service=MagicMock(),
        embeddings_service=MagicMock(),
    )


class TestParallelProcessing:
    """Test that fully parallel processing works when no rate limits occur."""

    @pytest.mark.asyncio
    async def test_parallel_processes_all_videos(self):
        """With a high concurrency limit, all videos are processed in
        the first (parallel) strategy pass and the job completes."""
        manager = _create_job_manager()
        job = manager.create_job(["vid1", "vid2", "vid3", "vid4", "vid5"])

        # Allow unlimited concurrency — parallel strategy succeeds
        tracker = ConcurrencyTracker(max_concurrent=100)
        manager._process_video = tracker.mock_process_video

        with patch.object(manager, '_process_parallel', wraps=manager._process_parallel) as spy_parallel, \
             patch.object(manager, '_process_bounded', wraps=manager._process_bounded) as spy_bounded, \
             patch.object(manager, '_process_sequential', wraps=manager._process_sequential) as spy_seq:
            await manager._process_job(job)

            # Verify only parallel was used
            assert spy_parallel.call_count == 1
            assert spy_bounded.call_count == 0
            assert spy_seq.call_count == 0

        assert job.status == JobStatus.COMPLETED
        assert job.videos_processed == 5
        assert all(job.videos[vid].indexed for vid in job.video_ids)


class TestBoundedParallelFallback:
    """Test that bounded parallelism is used when parallel hits rate limits."""

    @pytest.mark.asyncio
    async def test_falls_back_to_bounded_on_rate_limit(self):
        """With max_concurrent=3 and 5 videos, parallel (5 concurrent)
        fails for 2 videos. The bounded strategy (semaphore of 3) retries
        only the failed videos and succeeds."""
        manager = _create_job_manager()
        job = manager.create_job(["vid1", "vid2", "vid3", "vid4", "vid5"])

        tracker = ConcurrencyTracker(max_concurrent=3)
        manager._process_video = tracker.mock_process_video

        with patch.object(manager, '_process_parallel', wraps=manager._process_parallel) as spy_parallel, \
             patch.object(manager, '_process_bounded', wraps=manager._process_bounded) as spy_bounded, \
             patch.object(manager, '_process_sequential', wraps=manager._process_sequential) as spy_seq:
            await manager._process_job(job)

            # Verify parallel was tried first, then bounded took over
            assert spy_parallel.call_count == 1
            assert spy_bounded.call_count == 1
            assert spy_seq.call_count == 0

        assert job.status == JobStatus.COMPLETED
        assert job.videos_processed == 5
        assert all(job.videos[vid].indexed for vid in job.video_ids)


class TestSequentialFallback:
    """Test that sequential processing is used when bounded also hits limits."""

    @pytest.mark.asyncio
    async def test_falls_back_to_sequential_on_rate_limit(self):
        """With max_concurrent=1, both parallel and bounded fail.
        Sequential (1 at a time) succeeds."""
        manager = _create_job_manager()
        job = manager.create_job(["vid1", "vid2", "vid3"])

        tracker = ConcurrencyTracker(max_concurrent=1)
        manager._process_video = tracker.mock_process_video

        with patch.object(manager, '_process_parallel', wraps=manager._process_parallel) as spy_parallel, \
             patch.object(manager, '_process_bounded', wraps=manager._process_bounded) as spy_bounded, \
             patch.object(manager, '_process_sequential', wraps=manager._process_sequential) as spy_seq:
            await manager._process_job(job)

            # Verify all three strategies were tried in order
            assert spy_parallel.call_count == 1
            assert spy_bounded.call_count == 1
            assert spy_seq.call_count == 1

        assert job.status == JobStatus.COMPLETED
        assert job.videos_processed == 3
        assert all(job.videos[vid].indexed for vid in job.video_ids)


class TestIsRateLimitError:
    """Verify _is_rate_limit_error detects the right exception types."""

    def test_detects_video_processor_rate_limit(self):
        error = RateLimitError("rate limited", retry_after=30)
        assert _is_rate_limit_error(error) is True

    def test_detects_wrapped_rate_limit(self):
        """EmbeddingGenerationError wrapping a RateLimitError via 'from'."""
        cause = RateLimitError("rate limited")
        wrapper = RuntimeError("embedding failed")
        wrapper.__cause__ = cause
        assert _is_rate_limit_error(wrapper) is True

    def test_rejects_unrelated_errors(self):
        assert _is_rate_limit_error(ValueError("bad value")) is False
        assert _is_rate_limit_error(RuntimeError("something broke")) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
