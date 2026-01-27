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

    async def start_job(self, job_id: str):
        """Start processing a job in the background."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job {job_id} is already {job.status}")

        task = asyncio.create_task(self._process_job(job))
        self._running_tasks[job_id] = task

    async def _process_job(self, job: Job):
        """Process all videos in a job."""
        job.status = JobStatus.PROCESSING
        job.message = "Processing videos..."

        try:
            for video_id in job.video_ids:
                try:
                    await self._process_video(job, video_id)
                except Exception as e:
                    logger.error(f"Error processing video {video_id}: {e}")
                    # Continue with other videos
                    if video_id in job.videos:
                        job.videos[video_id].indexed = False

                job.videos_processed += 1
                job.message = f"Processed {job.videos_processed}/{len(job.video_ids)} videos"

            job.status = JobStatus.COMPLETED
            job.message = f"Successfully processed {job.videos_processed} videos"
            job.completed_at = datetime.utcnow()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.message = f"Job failed: {e}"
            logger.error(f"Job {job.job_id} failed: {e}")

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
