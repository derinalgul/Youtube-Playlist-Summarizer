# YouTube Knowledge Base

Paste a YouTube playlist, ask questions in plain English, and get answers that cite the exact moment in the video. Click a citation and the embedded player jumps to that timestamp.

**Stack:** FastAPI · ChromaDB · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) · React + Vite · pytest (67 tests)

<!-- Add a GIF here: paste a playlist → ask a question → click a citation → player jumps -->
<!-- ![demo](docs/demo.gif) -->

## How it works

```
Playlist URL
   │
   ▼
Video Processor ──► Transcription ──► Chunking + Embeddings ──► ChromaDB
 (yt-dlp, cache)    (manual captions      (timestamped chunks,
                     first, auto as        500 chars, 50 overlap)
                     fallback)                                     │
                                                                   ▼
Question ──► embed ──► top-5 chunk retrieval ──► gpt-4o-mini ──► Answer + timestamp citations
```

Indexing runs as background jobs, so the UI shows per-video progress while a playlist is processed, and you can start asking questions as soon as videos finish.

## Design decisions

- **Chunks are built from whole caption segments.** Each chunk knows exactly when it starts and ends in the video, which is what makes citations clickable instead of approximate.
- **Small chunks (500 characters, 50 overlap).** This favors precise search and tight 15 to 30 second citation windows over long context per chunk. The overlap keeps ideas that straddle a boundary findable. Both are configurable through `YKB_CHUNK_SIZE` and `YKB_OVERLAP`. See [docs/repo-analysis.md](docs/repo-analysis.md) for the tradeoffs.
- **Manual captions first, auto-generated as fallback.** Manual captions are more accurate, and auto captions cover roughly 95% of videos, so almost any playlist can be indexed without paying for audio transcription.
- **Rate limit aware processing.** Jobs start fully parallel. If YouTube, yt-dlp or OpenAI returns a rate limit (including errors wrapped inside other exceptions), the job falls back to bounded parallelism (3 at a time), then to sequential. Already indexed videos are never reprocessed. Other errors are handled per video without slowing the whole job down.
- **Caching with TTLs.** Metadata, captions and playlists are cached (7, 30 and 1 days), so resubmitting a playlist doesn't refetch or re-embed what's already there.
- **Bounded conversation history.** Follow-up questions keep the last 10 messages for context without letting prompts grow without limit.

## Project structure

```
backend/
  api/        FastAPI app, routes, request/response schemas, background job manager
  services/   video_processor, transcription, embeddings (chunking + ChromaDB), query_engine
  models/     caption and video data models
  config/     settings (env vars prefixed with YKB_)
frontend/     React app: submit form, job status, chat, video player
tests/        pytest suite for every service, the API and the job fallback strategies
docs/         design notes
```

## Running locally

**Prerequisites:** [uv](https://docs.astral.sh/uv/), Node.js 18+, an OpenAI API key.

```bash
# Backend
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
echo "OPENAI_API_KEY=your_api_key_here" > .env
python -m uvicorn backend.api.app:app --reload     # API docs at http://localhost:8000/docs

# Frontend (new terminal)
cd frontend && npm install && npm run dev          # app at http://localhost:3000

# Tests (venv active)
python -m pytest tests/ -v
```

<details>
<summary>Virtual environment tips</summary>

- Activate in every new terminal with `source .venv/bin/activate`, leave with `deactivate`.
- Or skip activation and prefix commands with `uv run`, for example `uv run uvicorn backend.api.app:app --reload`.
- Add a dependency: put it in `requirements.txt`, then `uv pip install -r requirements.txt`.
- Rebuild a broken venv: `rm -rf .venv && uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt`.
- VS Code: run *Python: Select Interpreter* and pick `./.venv/bin/python`.

</details>
