"""Background job management for video processing."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from backend.api.schemas import JobStatus, VideoInfo
from backend.services.embeddings import EmbeddingsService
from backend.services.transcription import TranscriptionService, NoCaptionsAvailableError
from backend.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class RateLimitDetected(Exception):
    """Raised internally when a rate limit error is detected during processing."""

    pass


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is caused by API rate limiting.

    Checks OpenAI, YouTube transcript API, yt-dlp, and the custom
    VideoProcessor RateLimitError. Also checks wrapped causes
    (e.g. EmbeddingGenerationError wrapping an OpenAI RateLimitError).
    """
    # OpenAI rate limit (HTTP 429)
    try:
        from openai import RateLimitError as OpenAIRateLimitError

        if isinstance(error, OpenAIRateLimitError):
            return True
    except ImportError:
        pass

    # YouTube transcript API blocking
    try:
        from youtube_transcript_api._errors import RequestBlocked

        if isinstance(error, RequestBlocked):
            return True
    except ImportError:
        pass

    # Custom video processor rate limit
    from backend.services.video_processor import RateLimitError as VideoRateLimitError

    if isinstance(error, VideoRateLimitError):
        return True

    # yt-dlp rate limit (embedded in DownloadError message)
    try:
        from yt_dlp.utils import DownloadError

        if isinstance(error, DownloadError):
            msg = str(error).lower()
            if "429" in msg or "rate limit" in msg or "too many requests" in msg:
                return True
    except ImportError:
        pass

    # Check wrapped cause (e.g. EmbeddingGenerationError wrapping OpenAI RateLimitError)
    if error.__cause__ is not None:
        return _is_rate_limit_error(error.__cause__)

    return False


class Job:
    """Represents a video processing job."""

    def __init__(self, job_id: str, video_ids: List[str]):
        self.job_id = job_id
        self.video_ids = video_ids
        self.status = JobStatus.PENDING
        self.videos: Dict[str, VideoInfo] = {}
        self.videos_processed = 0
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.message = "Job created"

    @property
    def progress(self) -> float:
        if not self.video_ids:
            return 100.0
        return (self.videos_processed / len(self.video_ids)) * 100

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "videos_total": len(self.video_ids),
            "videos_processed": self.videos_processed,
            "videos": list(self.videos.values()),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class JobManager:
    """Manages background video processing jobs."""

    def __init__(
        self,
        video_processor: VideoProcessor,
        transcription_service: TranscriptionService,
        embeddings_service: EmbeddingsService,
    ):
        self._video_processor = video_processor
        self._transcription = transcription_service
        self._embeddings = embeddings_service
        self._jobs: Dict[str, Job] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}

    def create_job(self, video_ids: List[str]) -> Job:
        """Create a new processing job."""
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id, video_ids)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        """List all jobs."""
        return list(self._jobs.values())

    def clear_jobs(self):
        """Clear all jobs and running tasks."""
        self._jobs.clear()
        self._running_tasks.clear()

    async def start_job(self, job_id: str):
        """Start processing a job in the background."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job {job_id} is already {job.status}")

        task = asyncio.create_task(self._process_job(job))
        self._running_tasks[job_id] = task

    def _get_unprocessed_video_ids(self, job: Job) -> List[str]:
        """Get video IDs that haven't been successfully indexed."""
        return [
            vid for vid in job.video_ids
            if vid not in job.videos or not job.videos[vid].indexed
        ]

    async def _process_job(self, job: Job):
        """Process all videos, falling back through strategies on rate limits.

        Tries fully parallel first, then bounded parallelism (semaphore of 3),
        then sequential. Only falls back when rate limit errors are detected;
        other errors are handled per-video without switching strategy.
        """
        job.status = JobStatus.PROCESSING
        job.message = "Processing videos..."

        strategies = [
            ("parallel", self._process_parallel),
            ("bounded parallel", self._process_bounded),
            ("sequential", self._process_sequential),
        ]

        try:
            for strategy_name, strategy_fn in strategies:
                remaining = self._get_unprocessed_video_ids(job)
                if not remaining:
                    break

                try:
                    logger.info(
                        f"Job {job.job_id}: trying {strategy_name} strategy "
                        f"for {len(remaining)} video(s)"
                    )
                    job.message = f"Processing videos ({strategy_name})..."
                    await strategy_fn(job, remaining)
                    break  # Success — no rate limit detected
                except RateLimitDetected as e:
                    logger.warning(
                        f"Job {job.job_id}: {strategy_name} hit rate limit: {e}. "
                        f"Falling back to next strategy."
                    )
                    continue
            else:
                # All strategies exhausted due to rate limits
                job.status = JobStatus.FAILED
                job.error = "All processing strategies exhausted due to rate limits"
                job.message = "Job failed: rate limited by external APIs"
                logger.error(f"Job {job.job_id}: all strategies exhausted")
                return

            job.status = JobStatus.COMPLETED
            job.message = f"Successfully processed {job.videos_processed} videos"
            job.completed_at = datetime.utcnow()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.message = f"Job failed: {e}"
            logger.error(f"Job {job.job_id} failed: {e}")

    async def _process_parallel(self, job: Job, video_ids: List[str]):
        """Process videos fully in parallel using asyncio.gather."""

        async def _safe_process(video_id: str):
            try:
                await self._process_video(job, video_id)
                return (video_id, None)
            except Exception as e:
                return (video_id, e)

        tasks = [_safe_process(vid) for vid in video_ids]
        results = await asyncio.gather(*tasks)

        rate_limit_errors = []
        for video_id, error in results:
            if error is None:
                job.videos_processed += 1
            elif _is_rate_limit_error(error):
                rate_limit_errors.append((video_id, error))
            else:
                job.videos_processed += 1
                logger.error(f"Error processing video {video_id}: {error}")
                if video_id in job.videos:
                    job.videos[video_id].indexed = False

        job.message = f"Processed {job.videos_processed}/{len(job.video_ids)} videos"

        if rate_limit_errors:
            raise RateLimitDetected(
                f"Rate limit hit for {len(rate_limit_errors)} video(s) "
                f"during parallel processing"
            )

    async def _process_bounded(self, job: Job, video_ids: List[str]):
        """Process videos with bounded parallelism (max 3 concurrent)."""
        semaphore = asyncio.Semaphore(3)
        rate_limit_hit = False

        async def _bounded_process(video_id: str):
            nonlocal rate_limit_hit
            if rate_limit_hit:
                return

            async with semaphore:
                if rate_limit_hit:
                    return

                try:
                    await self._process_video(job, video_id)
                    job.videos_processed += 1
                    job.message = (
                        f"Processed {job.videos_processed}/{len(job.video_ids)} videos"
                    )
                except Exception as e:
                    if _is_rate_limit_error(e):
                        rate_limit_hit = True
                        return
                    job.videos_processed += 1
                    logger.error(f"Error processing video {video_id}: {e}")
                    if video_id in job.videos:
                        job.videos[video_id].indexed = False

        tasks = [_bounded_process(vid) for vid in video_ids]
        await asyncio.gather(*tasks)

        if rate_limit_hit:
            raise RateLimitDetected(
                "Rate limit hit during bounded parallel processing"
            )

    async def _process_sequential(self, job: Job, video_ids: List[str]):
        """Process videos one at a time (original approach)."""
        for video_id in video_ids:
            try:
                await self._process_video(job, video_id)
            except Exception as e:
                if _is_rate_limit_error(e):
                    raise RateLimitDetected(
                        f"Rate limit hit during sequential processing: {e}"
                    )
                logger.error(f"Error processing video {video_id}: {e}")
                if video_id in job.videos:
                    job.videos[video_id].indexed = False

            job.videos_processed += 1
            job.message = (
                f"Processed {job.videos_processed}/{len(job.video_ids)} videos"
            )

    async def _process_video(self, job: Job, video_id: str):
        """Process a single video."""
        # Get metadata
        metadata = await asyncio.to_thread(
            self._video_processor.get_video_metadata, video_id
        )

        video_info = VideoInfo(
            video_id=video_id,
            title=metadata.title,
            channel=metadata.channel_name,
            duration_seconds=metadata.duration_seconds,
            indexed=False,
            chunks_count=0,
        )
        job.videos[video_id] = video_info

        # Get transcript
        try:
            caption = await asyncio.to_thread(
                self._transcription.get_transcript, video_id
            )
        except NoCaptionsAvailableError:
            logger.warning(f"No captions available for {video_id}")
            return

        # Index in vector database
        chunks_count = await asyncio.to_thread(
            self._embeddings.index_video,
            caption,
            metadata.title,
        )

        video_info.indexed = True
        video_info.chunks_count = chunks_count


# Global job manager instance (initialized by app startup)
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    if _job_manager is None:
        raise RuntimeError("JobManager not initialized. Call init_job_manager first.")
    return _job_manager


def init_job_manager(
    video_processor: VideoProcessor,
    transcription_service: TranscriptionService,
    embeddings_service: EmbeddingsService,
) -> JobManager:
    """Initialize the global job manager."""
    global _job_manager
    _job_manager = JobManager(
        video_processor, transcription_service, embeddings_service
    )
    return _job_manager
