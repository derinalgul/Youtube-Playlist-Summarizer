"""API module for YouTube Knowledge Base."""

from backend.api.app import app, create_app
from backend.api.jobs import JobManager, get_job_manager, init_job_manager

__all__ = [
    "app",
    "create_app",
    "JobManager",
    "get_job_manager",
    "init_job_manager",
]
