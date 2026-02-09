"""API routes for YouTube Knowledge Base."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Depends

from backend.api.schemas import (
    CitationResponse,
    CollectionStatsResponse,
    HealthResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    VideoInfo,
    VideoSubmitRequest,
)
from backend.api.jobs import get_job_manager, JobManager
from backend.models.video import VideoSource
from backend.services.embeddings import EmbeddingsService
from backend.services.query_engine import QueryEngine, NoResultsError
from backend.services.video_processor import VideoProcessor, URLParsingError

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Dependencies ==============


def get_video_processor() -> VideoProcessor:
    """Dependency to get VideoProcessor instance."""
    from backend.api.app import get_services
    return get_services()["video_processor"]


def get_embeddings_service() -> EmbeddingsService:
    """Dependency to get EmbeddingsService instance."""
    from backend.api.app import get_services
    return get_services()["embeddings"]


def get_query_engine() -> QueryEngine:
    """Dependency to get QueryEngine instance."""
    from backend.api.app import get_services
    return get_services()["query_engine"]


# ============== Health Check ==============


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={
            "video_processor": "ok",
            "embeddings": "ok",
            "query_engine": "ok",
        },
    )


# ============== Video/Playlist Submission ==============


@router.post("/videos", response_model=JobResponse, tags=["Videos"])
async def submit_video(
    request: VideoSubmitRequest,
    video_processor: VideoProcessor = Depends(get_video_processor),
    job_manager: JobManager = Depends(get_job_manager),
):
    """
    Submit a YouTube video or playlist for processing.

    The video(s) will be processed in the background:
    1. Extract metadata
    2. Download captions
    3. Generate embeddings
    4. Index in vector database

    Returns a job ID that can be used to check processing status.
    """
    try:
        # Parse the URL to get video ID(s)
        parsed = video_processor.parse_url(request.url)

        # Check if URL is valid
        if not parsed.is_valid:
            raise URLParsingError(parsed.error_message or "Invalid YouTube URL")

        if parsed.source_type == VideoSource.PLAYLIST:
            # Get playlist metadata to get video IDs
            playlist_meta = video_processor.get_playlist_metadata(parsed.playlist_id)
            video_ids = playlist_meta.video_ids
            message = f"Playlist submitted with {len(video_ids)} videos"
        else:
            video_ids = [parsed.video_id]
            message = "Video submitted for processing"

        # Create a job
        job = job_manager.create_job(video_ids)

        # Start processing in background
        await job_manager.start_job(job.job_id)

        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            message=message,
            videos=[],
            created_at=job.created_at,
        )

    except URLParsingError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YouTube URL: {e}")
    except Exception as e:
        logger.error(f"Error submitting video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
):
    """Get the status of a processing job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(**job.to_dict())


@router.get("/jobs", response_model=List[JobStatusResponse], tags=["Jobs"])
async def list_jobs(
    job_manager: JobManager = Depends(get_job_manager),
):
    """List all processing jobs."""
    jobs = job_manager.list_jobs()
    return [JobStatusResponse(**job.to_dict()) for job in jobs]


# ============== Query ==============


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_knowledge_base(
    request: QueryRequest,
    query_engine: QueryEngine = Depends(get_query_engine),
):
    """
    Ask a question about the indexed videos.

    The system will:
    1. Find relevant transcript chunks using semantic search
    2. Use an LLM to generate an answer based on the context
    3. Return the answer with citations to specific video timestamps
    """
    try:
        response = query_engine.query(
            question=request.question,
            video_ids=request.video_ids,
        )

        # Convert citations to response format
        citations = [
            CitationResponse(
                video_id=c.video_id,
                video_title=c.video_title,
                timestamp=c.format_timestamp(),
                timestamp_seconds=c.start_time,
                youtube_link=c.format_link(),
                text_snippet=c.text_snippet,
            )
            for c in response.citations
        ]

        return QueryResponse(
            answer=response.answer,
            citations=citations,
            model_used=response.model_used,
            tokens_used=response.tokens_used,
        )

    except NoResultsError as e:
        raise HTTPException(
            status_code=404,
            detail="No indexed content found. Please submit and process some videos first.",
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/query/history", tags=["Query"])
async def clear_conversation_history(
    query_engine: QueryEngine = Depends(get_query_engine),
):
    """Clear the conversation history for follow-up questions."""
    query_engine.clear_history()
    return {"message": "Conversation history cleared"}


# ============== Cache Management ==============


@router.delete("/cache", tags=["Cache"])
async def clear_cache(
    video_processor: VideoProcessor = Depends(get_video_processor),
    embeddings: EmbeddingsService = Depends(get_embeddings_service),
    job_manager: JobManager = Depends(get_job_manager),
):
    """Clear all cached data (metadata, captions, and embeddings)."""
    try:
        metadata_cleared = video_processor.clear_cache()
        chunks_cleared = embeddings.clear_collection()
        job_manager.clear_jobs()

        return {
            "message": "Cache cleared successfully",
            "metadata_cleared": metadata_cleared,
            "chunks_cleared": chunks_cleared,
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Collection Stats ==============


@router.get("/stats", response_model=CollectionStatsResponse, tags=["Stats"])
async def get_collection_stats(
    embeddings: EmbeddingsService = Depends(get_embeddings_service),
    job_manager: JobManager = Depends(get_job_manager),
):
    """Get statistics about the indexed collection."""
    stats = embeddings.get_collection_stats()

    # Count unique indexed videos
    indexed_videos = set()
    for job in job_manager.list_jobs():
        for video_id, info in job.videos.items():
            if info.indexed:
                indexed_videos.add(video_id)

    return CollectionStatsResponse(
        total_chunks=stats["total_chunks"],
        total_videos=len(indexed_videos),
        collection_name=stats["collection_name"],
    )
