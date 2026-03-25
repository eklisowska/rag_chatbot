import glob
import logging
import os
import traceback
from typing import Generator

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.clients import (
    get_embedding_model,
    get_whisper_model,
    reset_chroma_collection,
)
from src.config import (
    AUDIO_EXTENSIONS,
    AUDIO_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_PATH,
)

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        logger.error(
            "Error extracting text from %s: %s\n%s",
            pdf_path, e, traceback.format_exc()
        )
        return ""


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio from an audio/video file using Whisper."""
    try:
        model = get_whisper_model()
        logger.info("Transcribing: %s", audio_path)
        segments, _ = model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        logger.error(
            "Error transcribing %s: %s\n%s",
            audio_path, e, traceback.format_exc()
        )
        return ""


def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "source": source, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]




def process_pdf_files() -> Generator[dict, None, None]:
    """Process all PDF files and yield chunks."""
    pdf_files = glob.glob(os.path.join(PDF_PATH, "*.pdf"))
    logger.info("Found %d PDF files", len(pdf_files))

    for pdf_file in pdf_files:
        logger.info("Processing PDF: %s", pdf_file)
        text = extract_pdf_text(pdf_file)
        if not text.strip():
            logger.warning("Skipping %s: no text extracted", pdf_file)
            continue
        chunks = chunk_text(text, os.path.basename(pdf_file))
        logger.info("Extracted %d chunks from %s", len(chunks), pdf_file)
        yield from chunks


def process_audio_files() -> Generator[dict, None, None]:
    """Process all audio/video files and yield chunks."""
    audio_files: list[str] = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(glob.glob(os.path.join(AUDIO_PATH, ext)))
    logger.info("Found %d audio/video files", len(audio_files))

    for audio_file in audio_files:
        logger.info("Processing audio: %s", audio_file)
        text = transcribe_audio(audio_file)
        if not text.strip():
            logger.warning("Skipping %s: no text transcribed", audio_file)
            continue
        chunks = chunk_text(text, os.path.basename(audio_file))
        logger.info("Extracted %d chunks from %s", len(chunks), audio_file)
        yield from chunks


def ingest_all() -> None:
    """Process all PDFs and audio/video files, store in ChromaDB."""
    logger.info("Starting ingestion...")

    embedding_model = get_embedding_model()

    all_chunks = list(process_pdf_files()) + list(process_audio_files())

    if not all_chunks:
        logger.warning("No documents found to ingest!")
        return

    logger.info("Total chunks to embed: %d", len(all_chunks))

    texts = [chunk["text"] for chunk in all_chunks]
    logger.info("Generating embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    collection = reset_chroma_collection()

    logger.info("Storing in ChromaDB...")
    collection.add(
        ids=[str(i) for i in range(len(all_chunks))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
            for chunk in all_chunks
        ],
    )

    logger.info("Ingestion complete! Stored %d chunks.", len(all_chunks))


if __name__ == "__main__":
    ingest_all()
