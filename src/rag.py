import logging
from typing import Generator

import google.generativeai as genai

from src.clients import get_chroma_collection, get_embedding_model
from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context.
Use ONLY the information from the context to answer the question.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""


def get_all_chunks(limit: int | None = None, offset: int = 0) -> dict:
    """Get all chunks from ChromaDB with optional pagination."""
    collection = get_chroma_collection()
    total_count = collection.count()
    
    if total_count == 0:
        return {"total": 0, "chunks": []}
    
    results = collection.get(
        include=["documents", "metadatas"],
        limit=limit,
        offset=offset,
    )
    
    chunks = []
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        chunks.append({
            "id": results["ids"][i],
            "text": doc,
            "metadata": meta,
        })
    
    return {
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "returned": len(chunks),
        "chunks": chunks,
    }


def get_relevant_chunks(question: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Retrieve relevant chunks from ChromaDB based on the question."""
    model = get_embedding_model()
    question_embedding = model.encode([question])[0]
    
    collection = get_chroma_collection()
    
    if collection.count() == 0:
        return []
    
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": results["distances"][0][i] if results["distances"] else None,
        })
    
    return chunks


def _build_context(context_chunks: list[dict]) -> str:
    """Build formatted context string from chunks."""
    return "\n\n---\n\n".join([
        f"[Source: {chunk['source']}]\n{chunk['text']}"
        for chunk in context_chunks
    ])


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build the full prompt with context and question."""
    context = _build_context(context_chunks)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def _get_gemini_model():
    """Get configured Gemini model instance."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Generate an answer using Google Gemini with the retrieved context."""
    prompt = _build_prompt(question, context_chunks)

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        return response.text
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return f"Error: {e}"
    except Exception as e:
        logger.error("Failed to generate answer: %s", e)
        return f"Error: Failed to generate answer: {e}"


def generate_answer_stream(question: str, context_chunks: list[dict]) -> Generator[str, None, None]:
    """Generate an answer using Google Gemini with streaming."""
    prompt = _build_prompt(question, context_chunks)

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        yield f"\n\n[Error: {e}]"
    except Exception as e:
        logger.error("Stream error: %s", e)
        yield f"\n\n[Error: {e}]"


def query(question: str) -> str:
    """Full RAG pipeline: retrieve context and generate answer."""
    chunks = get_relevant_chunks(question)
    
    if not chunks:
        return "No documents found in the knowledge base. Please run the ingestion first."
    
    return generate_answer(question, chunks)


def query_stream(question: str) -> Generator[str, None, None]:
    """Full RAG pipeline with streaming response."""
    chunks = get_relevant_chunks(question)

    if not chunks:
        yield "No documents found in the knowledge base. Please run the ingestion first."
        return

    yield from generate_answer_stream(question, chunks)


if __name__ == "__main__":
    print("RAG Chatbot - Type 'quit' to exit")
    print("-" * 40)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nAnswer:", end=" ")
        for token in query_stream(question):
            print(token, end="", flush=True)
        print()
