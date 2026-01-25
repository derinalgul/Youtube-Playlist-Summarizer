"""Video processor for YouTube Knowledge Base.

Handles URL parsing, metadata extraction, caption downloading,
and audio extraction for transcription.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import yt_dlp
from diskcache import Cache
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from backend.config.settings import settings
from backend.models.video import (
    CaptionSegment,
    CaptionType,
    PlaylistMetadata,
    ProcessedVideo,
    URLParseResult,
    VideoCaption,
    VideoMetadata,
    VideoSource,
)

logger = logging.getLogger(__name__)


# === EXCEPTIONS ===


class VideoProcessorError(Exception):
    """Base exception for video processor errors."""

    def __init__(self, message: str, video_id: Optional[str] = None):
        self.message = message
        self.video_id = video_id
        super().__init__(self.message)


class URLParsingError(VideoProcessorError):
    """Raised when URL parsing/validation fails."""

    pass


class MetadataExtractionError(VideoProcessorError):
    """Raised when metadata extraction fails."""

    pass


class CaptionExtractionError(VideoProcessorError):
    """Raised when caption extraction fails."""

    pass


class AudioExtractionError(VideoProcessorError):
    """Raised when audio extraction fails."""

    pass


class VideoUnavailableError(VideoProcessorError):
    """Raised when video is unavailable (private, deleted, etc.)."""

    pass


class RateLimitError(VideoProcessorError):
    """Raised when YouTube rate limits requests."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


# === MAIN CLASS ===


class VideoProcessor:
    """
    Main video processor for YouTube Knowledge Base.

    Handles URL parsing, metadata extraction, caption download,
    and audio extraction for transcription.
    """

    # YouTube URL patterns
    VIDEO_PATTERNS = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
    ]
    PLAYLIST_PATTERN = r"[?&]list=([a-zA-Z0-9_-]+)"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        audio_output_dir: Optional[Path] = None,
        preferred_languages: Optional[List[str]] = None,
    ):
        """
        Initialize the VideoProcessor.

        Args:
            cache_dir: Directory for cache storage.
            audio_output_dir: Directory for extracted audio files.
            preferred_languages: Priority list of language codes for captions.
        """
        self.cache_dir = cache_dir or settings.cache_dir / "video_processor"
        self.audio_output_dir = audio_output_dir or settings.audio_output_dir
        self.preferred_languages = preferred_languages or settings.preferred_languages

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache
        self._cache = Cache(str(self.cache_dir / "metadata"))

        # Initialize transcript API instance
        self._transcript_api = YouTubeTranscriptApi()

        # yt-dlp options for metadata extraction
        self._ydl_opts_metadata = {
            "quiet": settings.ytdlp_quiet,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
            "ignoreerrors": True,
        }

        # yt-dlp options for playlist extraction (flat)
        self._ydl_opts_playlist = {
            "quiet": settings.ytdlp_quiet,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
            "ignoreerrors": True,
        }

        # yt-dlp options for audio extraction
        self._ydl_opts_audio = {
            "quiet": settings.ytdlp_quiet,
            "no_warnings": True,
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": settings.audio_format,
                    "preferredquality": settings.audio_quality,
                }
            ],
            "outtmpl": str(self.audio_output_dir / "%(id)s.%(ext)s"),
        }

    # === URL PARSING METHODS ===

    def parse_url(self, url: str) -> URLParseResult:
        """
        Parse and validate a YouTube URL.

        Supports:
        - Single video URLs (youtube.com/watch?v=..., youtu.be/...)
        - Playlist URLs (youtube.com/playlist?list=...)
        - Video URLs with playlist context

        Args:
            url: YouTube URL to parse.

        Returns:
            URLParseResult with parsed components.
        """
        url = url.strip()

        # Extract video ID
        video_id = self._extract_video_id(url)

        # Extract playlist ID
        playlist_id = self._extract_playlist_id(url)

        # Determine source type
        if playlist_id and not video_id:
            # Pure playlist URL
            source_type = VideoSource.PLAYLIST
        elif video_id:
            # Single video (possibly with playlist context)
            source_type = VideoSource.SINGLE_VIDEO
        else:
            return URLParseResult(
                url=url,
                source_type=VideoSource.SINGLE_VIDEO,
                is_valid=False,
                error_message="Could not extract video or playlist ID from URL",
            )

        return URLParseResult(
            url=url,
            source_type=source_type,
            video_id=video_id,
            playlist_id=playlist_id,
            is_valid=True,
        )

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        for pattern in self.VIDEO_PATTERNS:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Also check query parameters
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc:
            query_params = parse_qs(parsed.query)
            if "v" in query_params:
                return query_params["v"][0]

        return None

    def _extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from URL."""
        match = re.search(self.PLAYLIST_PATTERN, url)
        if match:
            return match.group(1)
        return None

    # === METADATA EXTRACTION METHODS ===

    def get_video_metadata(
        self, video_id: str, use_cache: bool = True
    ) -> VideoMetadata:
        """
        Extract metadata for a single video.

        Args:
            video_id: YouTube video ID.
            use_cache: Whether to use cached data if available.

        Returns:
            VideoMetadata object with all available metadata.

        Raises:
            MetadataExtractionError: If extraction fails.
        """
        cache_key = self._cache_key(video_id, "metadata")

        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for metadata: {video_id}")
            return VideoMetadata.model_validate(self._cache[cache_key])

        # Extract with yt-dlp
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            raw_info = self._extract_metadata_with_ytdlp(url)
            metadata = self._parse_video_metadata(raw_info)

            # Cache the result
            self._cache.set(
                cache_key, metadata.model_dump(), expire=settings.metadata_cache_ttl
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata for {video_id}: {e}")
            raise MetadataExtractionError(
                f"Failed to extract metadata: {e}", video_id=video_id
            ) from e

    def get_playlist_metadata(
        self, playlist_id: str, use_cache: bool = True
    ) -> PlaylistMetadata:
        """
        Extract metadata for a playlist (without full video details).

        Args:
            playlist_id: YouTube playlist ID.
            use_cache: Whether to use cached data if available.

        Returns:
            PlaylistMetadata with list of video IDs.
        """
        cache_key = self._cache_key(playlist_id, "playlist")

        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for playlist: {playlist_id}")
            return PlaylistMetadata.model_validate(self._cache[cache_key])

        url = f"https://www.youtube.com/playlist?list={playlist_id}"

        try:
            with yt_dlp.YoutubeDL(self._ydl_opts_playlist) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise MetadataExtractionError("yt-dlp returned None for playlist")

            # Extract video IDs from entries
            entries = info.get("entries", [])
            video_ids = [
                entry["id"] for entry in entries if entry and entry.get("id")
            ]

            metadata = PlaylistMetadata(
                playlist_id=playlist_id,
                title=info.get("title", "Unknown Playlist"),
                description=info.get("description"),
                channel_name=info.get("channel", info.get("uploader", "Unknown")),
                channel_id=info.get("channel_id", ""),
                video_count=len(video_ids),
                thumbnail_url=info.get("thumbnail"),
                video_ids=video_ids,
            )

            # Cache the result
            self._cache.set(
                cache_key, metadata.model_dump(), expire=settings.playlist_cache_ttl
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract playlist metadata for {playlist_id}: {e}")
            raise MetadataExtractionError(
                f"Failed to extract playlist metadata: {e}"
            ) from e

    def get_playlist_videos(
        self,
        playlist_id: str,
        use_cache: bool = True,
    ) -> Generator[VideoMetadata, None, None]:
        """
        Generator that yields VideoMetadata for each video in playlist.

        Args:
            playlist_id: YouTube playlist ID.
            use_cache: Whether to use cached data.

        Yields:
            VideoMetadata for each video in the playlist.
        """
        playlist_meta = self.get_playlist_metadata(playlist_id, use_cache=use_cache)

        for video_id in playlist_meta.video_ids:
            try:
                yield self.get_video_metadata(video_id, use_cache=use_cache)
            except MetadataExtractionError as e:
                logger.warning(f"Skipping video {video_id}: {e}")
                continue

    def _extract_metadata_with_ytdlp(self, url: str) -> dict:
        """Use yt-dlp to extract raw metadata dictionary."""
        with yt_dlp.YoutubeDL(self._ydl_opts_metadata) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise MetadataExtractionError("yt-dlp returned None for info")
            return info

    def _parse_video_metadata(self, raw_info: dict) -> VideoMetadata:
        """Convert yt-dlp info dict to VideoMetadata model."""
        # Check for caption availability
        subtitles = raw_info.get("subtitles", {})
        automatic_captions = raw_info.get("automatic_captions", {})

        manual_langs = list(subtitles.keys())
        auto_langs = list(automatic_captions.keys())

        # Parse upload date
        upload_date = None
        if raw_info.get("upload_date"):
            try:
                upload_date = datetime.strptime(raw_info["upload_date"], "%Y%m%d")
            except ValueError:
                pass

        return VideoMetadata(
            video_id=raw_info["id"],
            title=raw_info["title"],
            description=raw_info.get("description"),
            channel_name=raw_info.get("channel", raw_info.get("uploader", "Unknown")),
            channel_id=raw_info.get("channel_id", ""),
            duration_seconds=raw_info.get("duration", 0) or 0,
            thumbnail_url=raw_info.get("thumbnail"),
            upload_date=upload_date,
            view_count=raw_info.get("view_count"),
            like_count=raw_info.get("like_count"),
            has_manual_captions=len(manual_langs) > 0,
            has_auto_captions=len(auto_langs) > 0,
            available_languages=list(set(manual_langs + auto_langs)),
        )

    # === CAPTION EXTRACTION METHODS ===

    def get_captions(
        self,
        video_id: str,
        use_cache: bool = True,
        prefer_manual: bool = True,
    ) -> Optional[VideoCaption]:
        """
        Get captions for a video, prioritizing manual over auto-generated.

        Priority order (when prefer_manual=True):
        1. Manual captions in preferred languages
        2. Auto-generated captions in preferred languages
        3. Manual captions in any available language
        4. Auto-generated captions in any available language

        Args:
            video_id: YouTube video ID.
            use_cache: Whether to use cached captions.
            prefer_manual: Whether to prefer manual over auto-generated.

        Returns:
            VideoCaption if available, None otherwise.
        """
        cache_key = self._cache_key(video_id, "captions")

        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for captions: {video_id}")
            return VideoCaption.model_validate(self._cache[cache_key])

        try:
            # Try to fetch transcript
            transcript_data, caption_type, language = self._fetch_transcript(
                video_id, self.preferred_languages, prefer_manual=prefer_manual
            )

            caption = self._parse_transcript_to_caption(
                video_id, transcript_data, caption_type, language
            )

            # Cache the result
            self._cache.set(
                cache_key, caption.model_dump(), expire=settings.captions_cache_ttl
            )

            return caption

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.info(f"No captions available for {video_id}: {e}")
            return None
        except VideoUnavailable as e:
            logger.error(f"Video unavailable: {video_id}: {e}")
            raise CaptionExtractionError(
                f"Video unavailable: {e}", video_id=video_id
            ) from e
        except Exception as e:
            logger.error(f"Failed to extract captions for {video_id}: {e}")
            raise CaptionExtractionError(
                f"Caption extraction failed: {e}", video_id=video_id
            ) from e

    def list_available_captions(self, video_id: str) -> Tuple[List[str], List[str]]:
        """
        List available caption languages.

        Args:
            video_id: YouTube video ID.

        Returns:
            Tuple of (manual_languages, auto_generated_languages).
        """
        try:
            transcript_list = self._transcript_api.list(video_id)

            manual_langs = []
            auto_langs = []

            for transcript in transcript_list:
                if transcript.is_generated:
                    auto_langs.append(transcript.language_code)
                else:
                    manual_langs.append(transcript.language_code)

            return manual_langs, auto_langs

        except Exception as e:
            logger.warning(f"Could not list captions for {video_id}: {e}")
            return [], []

    def _fetch_transcript(
        self,
        video_id: str,
        language_codes: List[str],
        prefer_manual: bool = True,
    ) -> Tuple[List[dict], CaptionType, str]:
        """
        Fetch transcript data using youtube-transcript-api.

        Returns:
            Tuple of (transcript_data, caption_type, language).
        """
        transcript_list = self._transcript_api.list(video_id)

        if prefer_manual:
            # Try manual first
            try:
                transcript = transcript_list.find_manually_created_transcript(
                    language_codes
                )
                data = transcript.fetch()
                # Convert FetchedTranscriptSnippet objects to dicts
                data_dicts = [{"text": s.text, "start": s.start, "duration": s.duration} for s in data]
                return data_dicts, CaptionType.MANUAL, transcript.language_code
            except NoTranscriptFound:
                pass

        # Try auto-generated
        try:
            transcript = transcript_list.find_generated_transcript(language_codes)
            data = transcript.fetch()
            data_dicts = [{"text": s.text, "start": s.start, "duration": s.duration} for s in data]
            return data_dicts, CaptionType.AUTO_GENERATED, transcript.language_code
        except NoTranscriptFound:
            pass

        # Try any available transcript (manual or auto in any language)
        for transcript in transcript_list:
            try:
                data = transcript.fetch()
                data_dicts = [{"text": s.text, "start": s.start, "duration": s.duration} for s in data]
                caption_type = (
                    CaptionType.MANUAL
                    if not transcript.is_generated
                    else CaptionType.AUTO_GENERATED
                )
                return data_dicts, caption_type, transcript.language_code
            except Exception:
                continue

        raise NoTranscriptFound(video_id, language_codes, transcript_list)

    def _parse_transcript_to_caption(
        self,
        video_id: str,
        transcript_data: List[dict],
        caption_type: CaptionType,
        language: str,
    ) -> VideoCaption:
        """Convert raw transcript data to VideoCaption model."""
        segments = [
            CaptionSegment(
                text=item["text"], start=item["start"], duration=item["duration"]
            )
            for item in transcript_data
        ]

        full_text = " ".join(seg.text for seg in segments)

        return VideoCaption(
            video_id=video_id,
            language=language,
            caption_type=caption_type,
            segments=segments,
            full_text=full_text,
        )

    # === AUDIO EXTRACTION METHODS ===

    def extract_audio(
        self,
        video_id: str,
        force_download: bool = False,
    ) -> Path:
        """
        Extract audio from video for transcription.

        Only needed when captions are unavailable.

        Args:
            video_id: YouTube video ID.
            force_download: Re-download even if cached.

        Returns:
            Path to extracted audio file.

        Raises:
            AudioExtractionError: If extraction fails.
        """
        expected_path = self.audio_output_dir / f"{video_id}.{settings.audio_format}"

        # Check if already downloaded
        if not force_download and expected_path.exists():
            logger.debug(f"Audio already cached: {video_id}")
            return expected_path

        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            with yt_dlp.YoutubeDL(self._ydl_opts_audio) as ydl:
                ydl.download([url])

            if expected_path.exists():
                return expected_path

            # yt-dlp might have used a different extension
            for ext in ["mp3", "m4a", "webm", "opus"]:
                alt_path = self.audio_output_dir / f"{video_id}.{ext}"
                if alt_path.exists():
                    return alt_path

            raise AudioExtractionError(
                f"Audio file not found after download", video_id=video_id
            )

        except yt_dlp.utils.DownloadError as e:
            logger.error(f"Failed to extract audio for {video_id}: {e}")
            raise AudioExtractionError(
                f"Audio extraction failed: {e}", video_id=video_id
            ) from e
        except Exception as e:
            logger.error(f"Failed to extract audio for {video_id}: {e}")
            raise AudioExtractionError(
                f"Audio extraction failed: {e}", video_id=video_id
            ) from e

    def has_cached_audio(self, video_id: str) -> bool:
        """Check if audio is already extracted and cached."""
        return self.get_audio_path(video_id) is not None

    def get_audio_path(self, video_id: str) -> Optional[Path]:
        """Get path to cached audio file if it exists."""
        for ext in ["mp3", "m4a", "webm", "opus"]:
            path = self.audio_output_dir / f"{video_id}.{ext}"
            if path.exists():
                return path
        return None

    # === CACHE MANAGEMENT METHODS ===

    def is_cached(self, video_id: str, cache_type: str = "metadata") -> bool:
        """
        Check if data is cached for a video.

        Args:
            video_id: YouTube video ID.
            cache_type: One of "metadata", "captions", "playlist".
        """
        cache_key = self._cache_key(video_id, cache_type)
        return cache_key in self._cache

    def clear_cache(
        self,
        video_id: Optional[str] = None,
        cache_type: Optional[str] = None,
    ) -> int:
        """
        Clear cached data.

        Args:
            video_id: Specific video to clear, or None for all.
            cache_type: Specific type to clear, or None for all.

        Returns:
            Number of cache entries cleared.
        """
        if video_id is None and cache_type is None:
            # Clear everything
            count = len(self._cache)
            self._cache.clear()
            return count

        if video_id and cache_type:
            # Clear specific entry
            cache_key = self._cache_key(video_id, cache_type)
            if cache_key in self._cache:
                del self._cache[cache_key]
                return 1
            return 0

        # Clear matching pattern
        count = 0
        keys_to_delete = []

        for key in self._cache:
            if video_id and video_id in str(key):
                keys_to_delete.append(key)
            elif cache_type and str(key).startswith(cache_type):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._cache[key]
            count += 1

        return count

    def get_cache_stats(self) -> dict:
        """Get cache statistics (size, entry count, etc.)."""
        return {
            "size_bytes": self._cache.volume(),
            "entry_count": len(self._cache),
            "directory": str(self.cache_dir),
        }

    def _cache_key(self, video_id: str, cache_type: str) -> str:
        """Generate cache key for a video and type."""
        return f"{cache_type}:{video_id}"

    # === HIGH-LEVEL PROCESSING METHODS ===

    def process_video(
        self,
        video_id: str,
        extract_audio_if_no_captions: bool = True,
        use_cache: bool = True,
    ) -> ProcessedVideo:
        """
        Full processing pipeline for a single video.

        1. Extract metadata
        2. Attempt caption extraction
        3. If no captions and extract_audio_if_no_captions, extract audio

        Args:
            video_id: YouTube video ID.
            extract_audio_if_no_captions: Download audio when captions unavailable.
            use_cache: Use cached data where available.

        Returns:
            ProcessedVideo with all available data.
        """
        try:
            # Step 1: Get metadata
            metadata = self.get_video_metadata(video_id, use_cache=use_cache)

            # Step 2: Try to get captions
            caption = None
            try:
                caption = self.get_captions(video_id, use_cache=use_cache)
            except CaptionExtractionError as e:
                logger.warning(f"Caption extraction failed: {e}")

            # Step 3: Extract audio if no captions and requested
            audio_path = None
            if caption is None and extract_audio_if_no_captions:
                try:
                    audio_path = str(self.extract_audio(video_id))
                except AudioExtractionError as e:
                    logger.warning(f"Could not extract audio: {e}")

            return ProcessedVideo(
                metadata=metadata,
                caption=caption,
                audio_path=audio_path,
                processing_status="completed",
            )

        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            return ProcessedVideo(
                metadata=VideoMetadata(
                    video_id=video_id,
                    title="Unknown",
                    channel_name="Unknown",
                    channel_id="",
                    duration_seconds=0,
                ),
                processing_status="failed",
                error_message=str(e),
            )

    def process_url(
        self,
        url: str,
        extract_audio_if_no_captions: bool = True,
        use_cache: bool = True,
    ) -> List[ProcessedVideo]:
        """
        Process a YouTube URL (single video or playlist).

        Automatically detects URL type and processes accordingly.

        Args:
            url: YouTube URL (video or playlist).
            extract_audio_if_no_captions: Download audio when captions unavailable.
            use_cache: Use cached data where available.

        Returns:
            List of ProcessedVideo objects.
        """
        parse_result = self.parse_url(url)

        if not parse_result.is_valid:
            raise URLParsingError(parse_result.error_message or "Invalid URL")

        results = []

        if parse_result.source_type == VideoSource.PLAYLIST and parse_result.playlist_id:
            # Process playlist
            playlist_meta = self.get_playlist_metadata(
                parse_result.playlist_id, use_cache=use_cache
            )

            for video_id in playlist_meta.video_ids:
                result = self.process_video(
                    video_id,
                    extract_audio_if_no_captions=extract_audio_if_no_captions,
                    use_cache=use_cache,
                )
                results.append(result)
        elif parse_result.video_id:
            # Single video
            result = self.process_video(
                parse_result.video_id,
                extract_audio_if_no_captions=extract_audio_if_no_captions,
                use_cache=use_cache,
            )
            results.append(result)

        return results

    # === UTILITY METHODS ===

    def close(self):
        """Clean up resources (close cache connections)."""
        self._cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
