# RAG Chatbot with Open WebUI

A RAG chatbot that lets you chat with your PDF documents and audio/video recordings. Uses Google Gemini for LLM and ChromaDB for vector storage.


## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Open WebUI    │────▶│    RAG API      │────▶│  Google Gemini  │
│   (Port 3000)   │     │   (Port 8000)   │     │      API        │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    ChromaDB     │
                       │ (Vector Store)  │
                       └─────────────────┘
```

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get an API key at [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Add Your Data

```bash
cp your_document.pdf data/pdf/
cp your_recording.mp4 data/audio/
```

Supported: PDF, mp4, mkv, avi, webm, mov, mp3, wav

### 3. Start Services

```bash
docker compose up -d
```

### 4. Ingest Data

```bash
docker compose exec rag-api python -m src.ingest
```

### 5. Chat

Open [http://localhost:3000](http://localhost:3000), then:

1. Create an account
2. Go to **Settings** → **Admin Settings** → **Connections**
3. Add OpenAI API: URL `http://rag-api:8000/v1`, Key `not-needed`
4. Select "rag-chatbot" model and start chatting

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat (supports streaming) |
| `/v1/models` | GET | List models |
| `/v1/chunks` | GET | Browse chunks (`?limit=10&offset=0`) |
| `/health` | GET | Health check |

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model to use |
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` for Q&A metrics |

RAG parameters (in `src/config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | `1500` | Chunk size (chars) |
| `CHUNK_OVERLAP` | `300` | Overlap between chunks |
| `TOP_K_RESULTS` | `10` | Chunks to retrieve |

> **Note:** These values were tuned based on experimentation. See [Configuration Experiment](#configuration-experiment) for details.

## Project Structure

```
rag_chatbot/
├── data/
│   ├── pdf/            # PDF documents
│   └── audio/          # Audio/video files
├── chroma_db/          # Vector database (auto-created)
├── src/
│   ├── config.py       # Configuration
│   ├── clients.py      # Singleton clients
│   ├── ingest.py       # Data ingestion
│   ├── rag.py          # RAG logic
│   └── api.py          # FastAPI endpoints
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Tech Stack

| Category | Technology |
|----------|------------|
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB |
| **API Framework** | FastAPI + Uvicorn |
| **PDF Processing** | PyMuPDF |
| **Audio Transcription** | Faster Whisper |
| **Text Chunking** | LangChain Text Splitters |
| **Frontend** | Open WebUI |
| **Containerization** | Docker Compose |


## Debug Logging

Set `LOG_LEVEL=DEBUG` in `.env` to enable detailed Q&A metrics for each query:

```
QUESTION:  user's query
RETRIEVAL:  number of chunks, distance scores, embedding and search time
SOURCES: source documents with chunk counts
CONTEXT: total context size sent to LLM
ANSWER: generated response
TIMING: generation and total processing time
```

## Configuration Experiment

The RAG parameters were tuned through a comparative experiment:

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| `CHUNK_SIZE` | 1000 | 1500 | +50% |
| `CHUNK_OVERLAP` | 200 | 300 | +50% |
| `TOP_K_RESULTS` | 5 | 10 | +100% |

**Key findings:**
- Larger chunks with more overlap prevent information gaps
- Retrieving more chunks improves answer completeness, even if average similarity scores are slightly lower
- Trade-off: longer generation times but more comprehensive answers

Full analysis with metrics and examples: [logs_conclusions.md](logs_conclusions.md)

## Troubleshooting

**No documents found** → Run ingestion: `docker compose exec rag-api python -m src.ingest`

**API key not set** → Create `.env` file with `GEMINI_API_KEY`, then `docker compose up -d`

**Open WebUI can't connect** → Use `http://rag-api:8000/v1` (Docker network name), not `localhost`

**Check logs** → `docker compose logs -f rag-api`
