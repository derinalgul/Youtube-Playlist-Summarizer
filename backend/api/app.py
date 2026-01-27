"""FastAPI application for YouTube Knowledge Base."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env file for OPENAI_API_KEY

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.api.jobs import init_job_manager, JobManager
from backend.services.embeddings import EmbeddingsService
from backend.services.query_engine import QueryEngine
from backend.services.transcription import TranscriptionService
from backend.services.video_processor import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global services (initialized on startup)
_services: dict = {}


def get_services() -> dict:
    """Get the global services dictionary."""
    return _services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global _services

    logger.info("Starting up YouTube Knowledge Base API...")

    # Initialize services
    video_processor = VideoProcessor()
    transcription_service = TranscriptionService(video_processor=video_processor)
    embeddings_service = EmbeddingsService()
    query_engine = QueryEngine(embeddings_service=embeddings_service)

    # Initialize job manager
    job_manager = init_job_manager(
        video_processor=video_processor,
        transcription_service=transcription_service,
        embeddings_service=embeddings_service,
    )

    # Store services globally
    _services["video_processor"] = video_processor
    _services["transcription"] = transcription_service
    _services["embeddings"] = embeddings_service
    _services["query_engine"] = query_engine
    _services["job_manager"] = job_manager

    logger.info("All services initialized successfully")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down YouTube Knowledge Base API...")
    embeddings_service.close()
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="YouTube Knowledge Base API",
        description="""
        Convert YouTube playlists into an intelligent, searchable knowledge base.

        ## Features
        - Submit YouTube videos or playlists for processing
        - Automatic caption extraction and embedding generation
        - Semantic search across all indexed content
        - Natural language Q&A with video timestamp citations

        ## Workflow
        1. **Submit** a video or playlist URL via POST /api/videos
        2. **Check** processing status via GET /api/jobs/{job_id}
        3. **Query** the knowledge base via POST /api/query
        """,
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api")

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
