"""API request and response schemas."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class JobStatus(str, Enum):
    """Status of a processing job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============== Video/Playlist Submission ==============


class VideoSubmitRequest(BaseModel):
    """Request to submit a video or playlist for processing."""

    url: str = Field(..., description="YouTube video or playlist URL")

    class Config:
        json_schema_extra = {
            "example": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        }


class VideoInfo(BaseModel):
    """Information about a processed video."""

    video_id: str
    title: str
    channel: str
    duration_seconds: int
    indexed: bool = False
    chunks_count: int = 0


class JobResponse(BaseModel):
    """Response after submitting a video/playlist."""

    job_id: str
    status: JobStatus
    message: str
    videos: List[VideoInfo] = []
    created_at: datetime


class JobStatusResponse(BaseModel):
    """Response for job status check."""

    job_id: str
    status: JobStatus
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    message: str
    videos_total: int = 0
    videos_processed: int = 0
    videos: List[VideoInfo] = []
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# ============== Query ==============


class QueryRequest(BaseModel):
    """Request to query the knowledge base."""

    question: str = Field(..., min_length=1, description="The question to ask")
    video_ids: Optional[List[str]] = Field(
        None, description="Optional list of video IDs to search within"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I create a variable in Python?",
                "top_k": 5,
            }
        }


class CitationResponse(BaseModel):
    """A citation in the query response."""

    video_id: str
    video_title: str
    timestamp: str
    timestamp_seconds: float
    youtube_link: str
    text_snippet: str


class QueryResponse(BaseModel):
    """Response from the query endpoint."""

    answer: str
    citations: List[CitationResponse]
    model_used: str
    tokens_used: Optional[int] = None


# ============== Collection Stats ==============


class CollectionStatsResponse(BaseModel):
    """Statistics about the indexed collection."""

    total_chunks: int
    total_videos: int
    collection_name: str


# ============== Health Check ==============


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: dict
