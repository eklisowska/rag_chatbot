"""
Shared singleton clients for the RAG chatbot.

This module provides centralized access to expensive resources like
embedding models, ChromaDB client/collection, and Whisper model.
"""

from __future__ import annotations

import logging

import chromadb
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

_embedding_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_chroma_collection: chromadb.Collection | None = None
_whisper_model: WhisperModel | None = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or initialize the ChromaDB client (singleton)."""
    global _chroma_client
    if _chroma_client is None:
        logger.info("Initializing ChromaDB client at: %s", CHROMA_PATH)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def get_chroma_collection() -> chromadb.Collection:
    """Get or initialize the ChromaDB collection (singleton).
    
    Handles stale collection references by detecting NotFoundError
    and refreshing the collection automatically.
    """
    global _chroma_collection
    if _chroma_collection is None:
        client = get_chroma_client()
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    else:
        try:
            _chroma_collection.count()
        except chromadb.errors.NotFoundError:
            logger.warning("Cached collection was invalidated, refreshing...")
            _chroma_collection = None
            return get_chroma_collection()
    return _chroma_collection


def reset_chroma_collection() -> chromadb.Collection:
    """Delete and recreate the collection. Returns the new collection."""
    global _chroma_collection
    client = get_chroma_client()
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
    except ValueError:
        pass
    
    _chroma_collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return _chroma_collection


def get_whisper_model() -> WhisperModel:
    """Get or initialize the Whisper model (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model: %s", WHISPER_MODEL)
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model
