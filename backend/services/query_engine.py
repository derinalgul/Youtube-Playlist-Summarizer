"""Query Engine for YouTube Knowledge Base.

Orchestrates semantic search and LLM responses to answer
questions based on indexed video transcripts.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel

from backend.services.embeddings import EmbeddingsService, SearchResult

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """A citation referencing a specific moment in a video."""

    video_id: str
    video_title: str
    start_time: float
    end_time: float
    text_snippet: str

    def format_timestamp(self) -> str:
        """Format start_time as MM:SS or HH:MM:SS."""
        total_seconds = int(self.start_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def format_link(self) -> str:
        """Format as a clickable YouTube link."""
        timestamp_seconds = int(self.start_time)
        return f"https://youtu.be/{self.video_id}?t={timestamp_seconds}"

    def format_reference(self) -> str:
        """Format as [Video Title @ MM:SS]."""
        return f"[{self.video_title} @ {self.format_timestamp()}]"


class QueryResponse(BaseModel):
    """Response from the query engine."""

    answer: str
    citations: List[Citation]
    model_used: str
    tokens_used: Optional[int] = None


@dataclass
class Message:
    """A message in the conversation history."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationHistory:
    """Manages conversation history for follow-up questions."""

    messages: List[Message] = field(default_factory=list)
    max_messages: int = 10  # Keep last N messages

    def add_user_message(self, content: str):
        """Add a user message."""
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.messages.append(Message(role="assistant", content=content))
        self._trim()

    def _trim(self):
        """Keep only the last max_messages."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages_for_api(self) -> List[dict]:
        """Convert to OpenAI API format."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self):
        """Clear conversation history."""
        self.messages = []


class QueryEngineError(Exception):
    """Base exception for query engine errors."""

    pass


class NoResultsError(QueryEngineError):
    """No relevant results found for query."""

    pass


class QueryEngine:
    """
    Engine for answering questions using indexed video transcripts.

    Combines semantic search (ChromaDB) with LLM responses (OpenAI)
    to answer questions with citations to specific video timestamps.

    Usage:
        engine = QueryEngine(embeddings_service)
        response = engine.query("How do I create a variable in Python?")
        print(response.answer)
        for citation in response.citations:
            print(citation.format_reference())
    """

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on YouTube video transcripts.

IMPORTANT RULES:
1. Only answer based on the provided transcript context
2. If the context doesn't contain enough information to answer, say so
3. Keep answers concise and direct
4. When referencing information, mention which video it comes from
5. Use natural language, don't just quote the transcripts verbatim

The user's question will be followed by relevant transcript excerpts with timestamps."""

    def __init__(
        self,
        embeddings_service: EmbeddingsService,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_context_chunks: int = 5,
    ):
        """
        Initialize the QueryEngine.

        Args:
            embeddings_service: EmbeddingsService for semantic search
            openai_api_key: OpenAI API key (uses env var if not provided)
            model: OpenAI model to use for responses
            max_context_chunks: Maximum transcript chunks to include in context
        """
        self._embeddings = embeddings_service
        self._openai = OpenAI(api_key=openai_api_key)
        self._model = model
        self._max_context_chunks = max_context_chunks
        self._conversation = ConversationHistory()

        logger.info(f"QueryEngine initialized with model '{model}'")

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """
        Build context string from search results.

        Args:
            search_results: List of SearchResult from semantic search

        Returns:
            Formatted context string with timestamps
        """
        if not search_results:
            return "No relevant transcript excerpts found."

        context_parts = ["Here are relevant excerpts from the video transcripts:\n"]

        for i, result in enumerate(search_results, 1):
            chunk = result.chunk
            title = result.video_title or "Unknown Video"

            # Format timestamp
            total_seconds = int(chunk.start_time)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            timestamp = f"{minutes}:{seconds:02d}"

            context_parts.append(
                f"[Excerpt {i}] From \"{title}\" at {timestamp}:\n"
                f"\"{chunk.text}\"\n"
            )

        return "\n".join(context_parts)

    def _create_citations(self, search_results: List[SearchResult]) -> List[Citation]:
        """
        Create citation objects from search results.

        Args:
            search_results: List of SearchResult from semantic search

        Returns:
            List of Citation objects
        """
        citations = []
        for result in search_results:
            chunk = result.chunk
            citations.append(
                Citation(
                    video_id=chunk.video_id,
                    video_title=result.video_title or "Unknown Video",
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    text_snippet=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                )
            )
        return citations

    def query(
        self,
        question: str,
        video_ids: Optional[List[str]] = None,
        include_history: bool = True,
    ) -> QueryResponse:
        """
        Answer a question using indexed video transcripts.

        Args:
            question: The user's question
            video_ids: Optional list of video IDs to search within
            include_history: Whether to include conversation history

        Returns:
            QueryResponse with answer and citations
        """
        # Search for relevant chunks
        search_results = self._embeddings.search(
            query=question,
            top_k=self._max_context_chunks,
            video_ids=video_ids,
        )

        if not search_results:
            raise NoResultsError(
                "No relevant content found in the indexed videos. "
                "Make sure videos are indexed before querying."
            )

        # Build context from search results
        context = self._build_context(search_results)

        # Build messages for API
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add conversation history if enabled
        if include_history and self._conversation.messages:
            messages.extend(self._conversation.get_messages_for_api())

        # Add current question with context
        user_message = f"{question}\n\n---\n\n{context}"
        messages.append({"role": "user", "content": user_message})

        # Call OpenAI API
        response = self._openai.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else None

        # Update conversation history
        self._conversation.add_user_message(question)
        self._conversation.add_assistant_message(answer)

        # Create citations
        citations = self._create_citations(search_results)

        logger.info(
            f"Query answered using {len(search_results)} chunks, "
            f"{tokens_used} tokens"
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            model_used=self._model,
            tokens_used=tokens_used,
        )

    def clear_history(self):
        """Clear conversation history."""
        self._conversation.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Message]:
        """Get current conversation history."""
        return self._conversation.messages.copy()
