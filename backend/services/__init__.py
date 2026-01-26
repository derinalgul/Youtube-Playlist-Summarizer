"""Services module for the YouTube Knowledge Base."""

from backend.services.video_processor import (
    AudioExtractionError,
    CaptionExtractionError,
    MetadataExtractionError,
    RateLimitError,
    URLParsingError,
    VideoProcessor,
    VideoProcessorError,
    VideoUnavailableError,
)
from backend.services.transcription import (
    NoCaptionsAvailableError,
    TranscriptionError,
    TranscriptionService,
)
from backend.services.embeddings import (
    ChunkingError,
    EmbeddingGenerationError,
    EmbeddingsError,
    EmbeddingsService,
    SearchResult,
    TextChunk,
    chunk_transcript,
)
from backend.services.query_engine import (
    Citation,
    ConversationHistory,
    Message,
    NoResultsError,
    QueryEngine,
    QueryEngineError,
    QueryResponse,
)

__all__ = [
    # Video Processor
    "AudioExtractionError",
    "CaptionExtractionError",
    "MetadataExtractionError",
    "RateLimitError",
    "URLParsingError",
    "VideoProcessor",
    "VideoProcessorError",
    "VideoUnavailableError",
    # Transcription Service
    "NoCaptionsAvailableError",
    "TranscriptionError",
    "TranscriptionService",
    # Embeddings Service
    "ChunkingError",
    "EmbeddingGenerationError",
    "EmbeddingsError",
    "EmbeddingsService",
    "SearchResult",
    "TextChunk",
    "chunk_transcript",
    # Query Engine
    "Citation",
    "ConversationHistory",
    "Message",
    "NoResultsError",
    "QueryEngine",
    "QueryEngineError",
    "QueryResponse",
]