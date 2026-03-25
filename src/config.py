import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME: str = "rag_documents"

DATA_PATH: str = os.getenv("DATA_PATH", "./data")
PDF_PATH: str = os.path.join(DATA_PATH, "pdf")
AUDIO_PATH: str = os.path.join(DATA_PATH, "audio")

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

WHISPER_MODEL: str = "base"

TOP_K_RESULTS: int = 5

AUDIO_EXTENSIONS: list[str] = ["*.mp4", "*.mkv", "*.avi", "*.webm", "*.mov", "*.mp3", "*.wav"]
