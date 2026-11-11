# YouTube Knowledge Base

## Overview
Build a web application that converts YouTube playlists into an intelligent, searchable knowledge base. Users paste a playlist URL, the system processes all videos (extracting captions or transcribing audio), and creates a semantic search interface where users can ask questions and receive answers with precise video timestamps as citations.

## Core Functionality
1. **Video Ingestion**: Accept YouTube playlist/video URLs, extract metadata, download captions
2. **Transcription Pipeline**: Extract YouTube captions (manual preferred, auto-generated fallback)
3. **Knowledge Extraction**: Chunk transcripts, generate embeddings, store in vector database
4. **Intelligent Query**: Natural language Q&A with LLM that cites specific video timestamps
5. **Interactive Interface**: Web UI with embedded video player that jumps to referenced timestamps

## Project Modules

### Module 1: Video Processor 
- YouTube URL parsing and validation
- Metadata extraction (title, duration, thumbnail, channel)
- Caption detection and downloading
- Cache management to avoid re-processing

### Module 2: Transcription Service 
- Manual caption preference (prioritized over auto-generated)
- Auto-generated caption fallback (~95% of videos have these)
- Timestamp alignment and formatting
- Transcript caching and retrieval

### Module 3: Vector Database & Embeddings 
- Text chunking strategy (preserve context + timestamps)
- Embedding generation (OpenAI text-embedding-3-small)
- Vector DB setup (ChromaDB)
- Semantic search functionality

### Module 4: LLM Query Engine 
- Question processing and embedding
- Relevant chunk retrieval
- Context assembly for LLM
- Answer synthesis with timestamp citations
- Conversation history management

### Module 5: API Backend 
- RESTful endpoints (submit playlist, query, get status)
- Background job processing (async video indexing)
- Request/response schemas with validation
- Error handling and validation

### Module 6: Frontend Interface 
- Playlist submission form
- Processing status dashboard
- Chat interface for questions
- Video player with timestamp navigation

## Folder Structure

```
|-- .env
|-- .gitignore
|-- .gitkeep
|-- .pytest_cache
    |-- .gitignore
    |-- CACHEDIR.TAG
    |-- README.md
    |-- v
|-- README.md
|-- backend
    |-- __init__.py
    |-- api
        |-- __init__.py
        |-- app.py
        |-- jobs.py
        |-- routes.py
        |-- schemas.py
    |-- config
        |-- __init__.py
        |-- settings.py
    |-- models
        |-- __init__.py
        |-- video.py
    |-- services
        |-- __init__.py
        |-- embeddings.py
        |-- query_engine.py
        |-- transcription.py
        |-- video_processor.py
|-- docs
    |-- .gitkeep
|-- frontend
    |-- index.html
    |-- package-lock.json
    |-- package.json
    |-- src
        |-- App.jsx
        |-- components
            |-- ChatInterface.jsx
            |-- JobStatus.jsx
            |-- Stats.jsx
            |-- VideoPlayer.jsx
            |-- VideoSubmitForm.jsx
        |-- index.css
        |-- main.jsx
        |-- services
            |-- api.js
    |-- vite.config.js
|-- requirements.txt
|-- scripts
    |-- .gitkeep
|-- tests
    |-- .gitkeep
    |-- test_api.py
    |-- test_embeddings.py
    |-- test_query_engine.py
    |-- test_transcription.py
    |-- test_video_processor.py
```

## How to Run

### Prerequisites
- [uv](https://docs.astral.sh/uv/) (installs Python and manages the virtual environment)
- Python 3.12 (installed via uv, see below)
- Node.js 18+
- OpenAI API key

### Setup

The backend runs inside a project-local virtual environment (`.venv/`), so its
packages stay isolated from your system Python and other projects. `.venv/` is
git-ignored; each machine creates its own.

1. **Install uv** (once per machine), with Homebrew:
   ```bash
   brew install uv
   ```
   or with the official installer:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python 3.12** (once per machine, can be run from any directory):
   ```bash
   uv python install 3.12
   ```

3. **Create the virtual environment** (run from the project root):
   ```bash
   uv venv --python 3.12
   ```

4. **Activate it and install Python dependencies:**
   ```bash
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

5. **Create `.env` file:**
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

6. **Install frontend dependencies** (then return to the project root):
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Using the Virtual Environment

All Python commands below (backend, tests) must run inside the venv.

- **Activate** (in every new terminal, from the project root):
  ```bash
  source .venv/bin/activate
  ```
  Your prompt shows `(Youtube Knowledge Base)` (uv names the venv after the project
  folder) while it's active, and `python` / `pip` point to the venv.
- **Deactivate** (leave the venv, back to system Python):
  ```bash
  deactivate
  ```
- **Run without activating:** prefix commands with `uv run`, e.g.
  `uv run uvicorn backend.api.app:app --reload`.
- **Add a new dependency:** add it to `requirements.txt`, then run
  `uv pip install -r requirements.txt` with the venv active.
- **Rebuild from scratch** (if the venv gets broken):
  ```bash
  deactivate 2>/dev/null; rm -rf .venv
  uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
  ```
- **VS Code:** run *Python: Select Interpreter* and choose `./.venv/bin/python`
  so the editor and test runner use the same packages.

### Running the Application

1. **Start the backend API** (from the project root, with the venv active):
   ```bash
   python -m uvicorn backend.api.app:app --reload
   ```
   Stop the server with `Ctrl+C`.

2. **Start the frontend (in a new terminal):**
   ```bash
   cd frontend
   npm run dev
   ```
   Stop it with `Ctrl+C`.

3. **Open browser:**
   - Frontend: http://localhost:3000
   - API docs: http://localhost:8000/docs

### Running Tests
With the venv active:
```bash
python -m pytest tests/ -v
```
