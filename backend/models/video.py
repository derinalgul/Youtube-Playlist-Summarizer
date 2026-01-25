"""Data models for video processing."""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class VideoSource(str, Enum):
    """Source type for the video URL."""
    SINGLE_VIDEO = "single_video"
    PLAYLIST = "playlist"


class CaptionType(str, Enum):
    """Type of caption source."""
    MANUAL = "manual"
    AUTO_GENERATED = "auto_generated"
    TRANSCRIBED = "transcribed"  # Via Whisper when no captions available
    NONE = "none"


class CaptionSegment(BaseModel):
    """A single caption segment with timestamp."""
    text: str
    start: float  # Start time in seconds
    duration: float  # Duration in seconds

    @property
    def end(self) -> float:
        """End time in seconds."""
        return self.start + self.duration


class VideoCaption(BaseModel):
    """Complete caption data for a video."""
    video_id: str
    language: str
    caption_type: CaptionType
    segments: List[CaptionSegment]
    full_text: str = ""  # Concatenated text for embedding
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class VideoMetadata(BaseModel):
    """Metadata for a single YouTube video."""
    video_id: str
    title: str
    description: Optional[str] = None
    channel_name: str
    channel_id: str
    duration_seconds: int
    thumbnail_url: Optional[HttpUrl] = None
    upload_date: Optional[datetime] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None

    # Caption availability
    has_manual_captions: bool = False
    has_auto_captions: bool = False
    available_languages: List[str] = Field(default_factory=list)

    # Processing state
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def duration_formatted(self) -> str:
        """Return duration as HH:MM:SS string."""
        return str(timedelta(seconds=self.duration_seconds))

    @property
    def youtube_url(self) -> str:
        """Return the YouTube watch URL."""
        return f"https://www.youtube.com/watch?v={self.video_id}"


class PlaylistMetadata(BaseModel):
    """Metadata for a YouTube playlist."""
    playlist_id: str
    title: str
    description: Optional[str] = None
    channel_name: str
    channel_id: str
    video_count: int
    thumbnail_url: Optional[HttpUrl] = None
    video_ids: List[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def youtube_url(self) -> str:
        """Return the YouTube playlist URL."""
        return f"https://www.youtube.com/playlist?list={self.playlist_id}"


class ProcessedVideo(BaseModel):
    """Complete processed video with metadata and captions."""
    metadata: VideoMetadata
    caption: Optional[VideoCaption] = None
    audio_path: Optional[str] = None  # Path to extracted audio if transcription needed
    processing_status: str = "pending"
    error_message: Optional[str] = None


class URLParseResult(BaseModel):
    """Result of parsing a YouTube URL."""
    url: str
    source_type: VideoSource
    video_id: Optional[str] = None
    playlist_id: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None
