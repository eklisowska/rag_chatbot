# RAG Chatbot with Open WebUI

A simple RAG (Retrieval-Augmented Generation) chatbot that processes PDF documents and audio/video recordings, with an Open WebUI interface. Uses Google Gemini API for LLM inference.

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

### 1. Get a Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Copy the example environment file and add your key:

```bash
cp .env.example .env
# Edit .env and replace 'your-api-key-here' with your actual API key
```

### 2. Add Your Data

Place your files in the data folders:

```bash
# PDF files
cp your_document.pdf data/pdf/

# Audio/video files (mp4, mkv, avi, webm, mov, mp3, wav)
cp your_video.mp4 data/audio/
```

### 3. Start Services

```bash
docker compose up -d
```

This starts two services:

- **RAG API** (port 8000): OpenAI-compatible RAG pipeline with Gemini
- **Open WebUI** (port 3000): Chat interface

### 4. Ingest Your Data

```bash
docker compose exec rag-api python -m src.ingest
```

This will:

- Extract text from all PDFs (using PyMuPDF)
- Transcribe audio/video files (using faster-whisper)
- Chunk text with overlap for context preservation
- Generate embeddings (all-MiniLM-L6-v2)
- Store everything in ChromaDB

### 5. Start Chatting

Open [http://localhost:3000](http://localhost:3000) in your browser.

**First-time Setup in Open WebUI:**

1. Create an account (stored locally)
2. Go to **Settings** (gear icon) → **Admin Settings** → **Connections**
3. Under "OpenAI API", add:
   - URL: `http://rag-api:8000/v1`
   - API Key: `not-needed`
4. Click "Save"
5. In the chat, select "rag-chatbot" model
6. Start asking questions about your documents!

## API Endpoints

The RAG API is OpenAI-compatible and exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (supports streaming) |
| `/v1/models` | GET | List available models |
| `/v1/chunks` | GET | List stored chunks (with pagination) |
| `/health` | GET | Health check with ChromaDB status |

## Project Structure

```
rag_chatbot/
├── data/
│   ├── pdf/            # Place PDF files here
│   └── audio/          # Place audio/video files here
├── chroma_db/          # Vector database (auto-created)
├── src/
│   ├── config.py       # Configuration settings
│   ├── clients.py      # Singleton clients (ChromaDB, embeddings, Whisper)
│   ├── ingest.py       # Data ingestion pipeline
│   ├── rag.py          # RAG retrieval and generation logic
│   └── api.py          # FastAPI OpenAI-compatible endpoints
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example        # Environment template
└── README.md
```

## Configuration

Edit `src/config.py` or use environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB storage path |
| `DATA_PATH` | `./data` | Base path for input data |
| `CHUNK_SIZE` | `1000` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |

### Available Gemini Models

| Model | Description |
|-------|-------------|
| `gemini-2.0-flash` | Fast, efficient model (default) |
| `gemini-2.0-flash-lite` | Lightweight, fastest response |
| `gemini-1.5-flash` | Balanced speed and quality |
| `gemini-1.5-pro` | Most capable, best for complex tasks |

## Supported File Formats

- **Documents**: PDF
- **Audio/Video**: mp4, mkv, avi, webm, mov, mp3, wav

## Troubleshooting

### "No documents found"

Run the ingestion: `docker compose exec rag-api python -m src.ingest`

### "GEMINI_API_KEY environment variable is not set"

Make sure you have a `.env` file with your API key:

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Then restart the services: `docker compose up -d`

### Gemini API rate limits

The free tier has rate limits (60 requests/minute). If you hit limits, consider:
- Using `gemini-2.0-flash-lite` for faster, lighter requests
- Upgrading to a paid plan for higher limits

### Audio/video transcription is slow

Whisper runs on CPU by default. The "base" model processes audio at approximately 1-2x real-time speed.

### Check service health

```bash
# Check all services
docker compose ps

# View logs
docker compose logs rag-api
docker compose logs open-webui

# Test API health
curl http://localhost:8000/health
```

### Connection issues

If Open WebUI can't connect to the RAG API, verify the connection URL uses the Docker network name (`http://rag-api:8000/v1`), not `localhost`.

## Technology Stack

- **PDF Extraction**: PyMuPDF (fitz)
- **Audio Transcription**: faster-whisper
- **Text Chunking**: LangChain text splitters
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (persistent storage)
- **LLM**: Google Gemini API (cloud inference)
- **API**: FastAPI (OpenAI-compatible)
- **UI**: Open WebUI

