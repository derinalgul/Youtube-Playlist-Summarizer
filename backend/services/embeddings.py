"""Embeddings service for YouTube Knowledge Base.

Provides text chunking, embedding generation, and semantic search
using OpenAI embeddings and ChromaDB vector database.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI
from pydantic import BaseModel

from backend.config.settings import get_settings
from backend.models.video import CaptionSegment, VideoCaption

logger = logging.getLogger(__name__)


class TextChunk(BaseModel):
    """A chunk of text with timestamp information."""

    text: str
    video_id: str
    start_time: float
    end_time: float
    chunk_index: int


class SearchResult(BaseModel):
    """A search result with relevance score."""

    chunk: TextChunk
    score: float
    video_title: Optional[str] = None


class EmbeddingsError(Exception):
    """Base exception for embeddings errors."""

    pass


class ChunkingError(EmbeddingsError):
    """Error during text chunking."""

    pass


class EmbeddingGenerationError(EmbeddingsError):
    """Error generating embeddings."""

    pass


def chunk_transcript(
    caption: VideoCaption,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[TextChunk]:
    """
    Chunk a video transcript while preserving timestamp information.

    Strategy:
    - Group caption segments into chunks of approximately `chunk_size` characters
    - Preserve timestamp boundaries (start/end times for each chunk)
    - Add overlap between chunks for better context retrieval

    Args:
        caption: VideoCaption with segments to chunk
        chunk_size: Target size in characters for each chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of TextChunk objects with preserved timestamps
    """
    if not caption.segments:
        return []

    chunks: List[TextChunk] = []
    current_text = ""
    current_start = caption.segments[0].start
    current_segments: List[CaptionSegment] = []
    chunk_index = 0

    for segment in caption.segments:
        # Add segment to current chunk
        current_segments.append(segment)
        if current_text:
            current_text += " " + segment.text
        else:
            current_text = segment.text
            current_start = segment.start

        # Check if chunk is large enough
        if len(current_text) >= chunk_size:
            # Calculate end time from last segment
            last_segment = current_segments[-1]
            end_time = last_segment.start + last_segment.duration

            chunks.append(
                TextChunk(
                    text=current_text.strip(),
                    video_id=caption.video_id,
                    start_time=current_start,
                    end_time=end_time,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            # Handle overlap - keep some segments for next chunk
            overlap_text = ""
            overlap_segments: List[CaptionSegment] = []

            # Work backwards to get overlap
            for seg in reversed(current_segments):
                test_text = seg.text + (" " + overlap_text if overlap_text else "")
                if len(test_text) <= chunk_overlap:
                    overlap_text = test_text
                    overlap_segments.insert(0, seg)
                else:
                    break

            current_text = overlap_text
            current_segments = overlap_segments
            if overlap_segments:
                current_start = overlap_segments[0].start

    # Don't forget the last chunk
    if current_text.strip():
        last_segment = current_segments[-1] if current_segments else caption.segments[-1]
        end_time = last_segment.start + last_segment.duration

        chunks.append(
            TextChunk(
                text=current_text.strip(),
                video_id=caption.video_id,
                start_time=current_start,
                end_time=end_time,
                chunk_index=chunk_index,
            )
        )

    return chunks


def generate_chunk_id(chunk: TextChunk) -> str:
    """Generate a unique ID for a chunk based on video_id and content."""
    content = f"{chunk.video_id}:{chunk.chunk_index}:{chunk.start_time}"
    return hashlib.md5(content.encode()).hexdigest()


class EmbeddingsService:
    """
    Service for generating embeddings and performing semantic search.

    Uses OpenAI's text-embedding-3-small model for embeddings and
    ChromaDB for vector storage and retrieval.

    Usage:
        service = EmbeddingsService()
        service.index_video(video_id, caption)
        results = service.search("What is Python?", top_k=5)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        persist_directory: Optional[Path] = None,
        collection_name: str = "youtube_transcripts",
    ):
        """
        Initialize the EmbeddingsService.

        Args:
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the Chroma collection
        """
        settings = get_settings()

        # Initialize OpenAI client
        self._openai = OpenAI(api_key=openai_api_key)
        self._model = "text-embedding-3-small"

        # Initialize ChromaDB
        persist_dir = persist_directory or settings.cache_dir / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._chunk_size = 500
        self._chunk_overlap = 50

        logger.info(
            f"EmbeddingsService initialized with collection '{collection_name}'"
        )

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self._openai.embeddings.create(
                model=self._model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to generate embedding: {e}") from e

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = self._openai.embeddings.create(
                model=self._model,
                input=texts,
            )
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings batch: {e}"
            ) from e

    def index_video(
        self,
        caption: VideoCaption,
        video_title: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Index a video transcript for semantic search.

        Chunks the transcript, generates embeddings, and stores in ChromaDB.

        Args:
            caption: VideoCaption to index
            video_title: Optional title for metadata
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap

        Returns:
            Number of chunks indexed
        """
        # Chunk the transcript
        chunks = chunk_transcript(
            caption,
            chunk_size=chunk_size or self._chunk_size,
            chunk_overlap=chunk_overlap or self._chunk_overlap,
        )

        if not chunks:
            logger.warning(f"No chunks generated for video {caption.video_id}")
            return 0

        # Generate embeddings in batches
        texts = [chunk.text for chunk in chunks]
        embeddings = self._get_embeddings_batch(texts)

        # Prepare data for ChromaDB
        ids = [generate_chunk_id(chunk) for chunk in chunks]
        metadatas = [
            {
                "video_id": chunk.video_id,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "chunk_index": chunk.chunk_index,
                "video_title": video_title or "",
            }
            for chunk in chunks
        ]

        # Upsert to ChromaDB
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(
            f"Indexed {len(chunks)} chunks for video {caption.video_id}"
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        video_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant chunks using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            video_ids: Optional list of video IDs to filter results

        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Build filter if video_ids specified
        where_filter = None
        if video_ids:
            where_filter = {"video_id": {"$in": video_ids}}

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results: List[SearchResult] = []

        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]
                distance = results["distances"][0][i]

                # Convert distance to similarity score (cosine distance to similarity)
                score = 1 - distance

                chunk = TextChunk(
                    text=document,
                    video_id=metadata["video_id"],
                    start_time=metadata["start_time"],
                    end_time=metadata["end_time"],
                    chunk_index=metadata["chunk_index"],
                )

                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=score,
                        video_title=metadata.get("video_title"),
                    )
                )

        return search_results

    def delete_video(self, video_id: str) -> int:
        """
        Delete all chunks for a video from the index.

        Args:
            video_id: Video ID to delete

        Returns:
            Number of chunks deleted
        """
        # Get all chunks for this video
        results = self._collection.get(
            where={"video_id": video_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            return 0

        # Delete them
        self._collection.delete(ids=results["ids"])

        logger.info(f"Deleted {len(results['ids'])} chunks for video {video_id}")
        return len(results["ids"])

    def clear_collection(self) -> int:
        """
        Delete all chunks from the collection.

        Returns:
            Number of chunks that were deleted
        """
        count = self._collection.count()
        if count == 0:
            return 0

        collection_name = self._collection.name
        collection_metadata = self._collection.metadata

        # Drop and recreate the collection
        self._chroma_client.delete_collection(collection_name)
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata,
        )

        logger.info(f"Cleared {count} chunks from collection '{collection_name}'")
        return count

    def is_video_indexed(self, video_id: str) -> bool:
        """
        Check if a video is already indexed.

        Args:
            video_id: Video ID to check

        Returns:
            True if video has indexed chunks
        """
        results = self._collection.get(
            where={"video_id": video_id},
            limit=1,
        )
        return len(results["ids"]) > 0

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current collection.

        Returns:
            Dict with count and other stats
        """
        return {
            "total_chunks": self._collection.count(),
            "collection_name": self._collection.name,
        }

    def close(self):
        """Clean up resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
