# YouTube Knowledge Base - Project Prompt

## Overview
Build a web application that converts YouTube playlists into an intelligent, searchable knowledge base. Users paste a playlist URL, the system processes all videos (extracting captions or transcribing audio), and creates a semantic search interface where users can ask questions and receive answers with precise video timestamps as citations.

## Core Functionality
1. **Video Ingestion**: Accept YouTube playlist/video URLs, extract metadata, download captions
2. **Transcription Pipeline**: Extract YouTube captions (manual preferred, auto-generated fallback)
3. **Knowledge Extraction**: Chunk transcripts, generate embeddings, store in vector database
4. **Intelligent Query**: Natural language Q&A with LLM that cites specific video timestamps
5. **Interactive Interface**: Web UI with embedded video player that jumps to referenced timestamps

## Project Modules

### Module 1: Video Processor ✅
- YouTube URL parsing and validation
- Metadata extraction (title, duration, thumbnail, channel)
- Caption detection and downloading
- Cache management to avoid re-processing

### Module 2: Transcription Service ✅
- Manual caption preference (prioritized over auto-generated)
- Auto-generated caption fallback (~95% of videos have these)
- Timestamp alignment and formatting
- Transcript caching and retrieval

### Module 3: Vector Database & Embeddings ✅
- Text chunking strategy (preserve context + timestamps)
- Embedding generation (OpenAI text-embedding-3-small)
- Vector DB setup (ChromaDB)
- Semantic search functionality

### Module 4: LLM Query Engine ✅
- Question processing and embedding
- Relevant chunk retrieval
- Context assembly for LLM
- Answer synthesis with timestamp citations
- Conversation history management

### Module 5: API Backend ✅
- RESTful endpoints (submit playlist, query, get status)
- Background job processing (async video indexing)
- Request/response schemas with validation
- Error handling and validation

### Module 6: Frontend Interface
- Playlist submission form
- Processing status dashboard
- Chat interface for questions
- Video player with timestamp navigation
- Transcript viewer with highlighting

## Folder Structure

```
youtube-knowledge-base/
├── backend/
│   ├── api/                 # FastAPI routes
│   ├── services/            # Core business logic modules
│   │   ├── video_processor.py
│   │   ├── transcription.py
│   │   ├── embeddings.py
│   │   └── llm_query.py
│   ├── models/              # Database models
│   ├── utils/               # Helper functions
│   └── config/              # Configuration files
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/        # API calls
│   │   └── utils/
├── tests/
├── docs/
└── scripts/                 # Setup/migration scripts
```

