import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Generator

import google.generativeai as genai

from src.clients import get_chroma_collection, get_embedding_model
from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)
qa_logger = logging.getLogger("rag.qa")

CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class RetrievalMetrics:
    """Metrics collected during chunk retrieval."""
    chunks: list[dict] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    embed_time_ms: float = 0.0
    search_time_ms: float = 0.0
    context_chars: int = 0

    @property
    def avg_distance(self) -> float:
        return sum(self.distances) / len(self.distances) if self.distances else 0.0

    @property
    def context_tokens_estimate(self) -> int:
        return self.context_chars // CHARS_PER_TOKEN_ESTIMATE

    def source_breakdown(self) -> str:
        counts = Counter(self.sources)
        return ", ".join(f"{src} ({cnt} chunks)" for src, cnt in counts.items())


def log_qa_interaction(
    question: str,
    answer: str,
    metrics: RetrievalMetrics,
    generation_time_ms: float,
    total_time_ms: float,
) -> None:
    """Log detailed Q&A interaction at DEBUG level."""
    if not qa_logger.isEnabledFor(logging.DEBUG):
        return

    separator = "═" * 70
    distances_str = ", ".join(f"{d:.3f}" for d in metrics.distances)

    qa_logger.debug(separator)
    qa_logger.debug("QUESTION: %s", question)
    qa_logger.debug(
        "RETRIEVAL: %d chunks | distances=[%s] | avg=%.3f | embed_ms=%.0f | search_ms=%.0f",
        len(metrics.chunks),
        distances_str,
        metrics.avg_distance,
        metrics.embed_time_ms,
        metrics.search_time_ms,
    )
    qa_logger.debug("SOURCES: %s", metrics.source_breakdown() or "none")
    qa_logger.debug(
        "CONTEXT: %d chars (~%d tokens)",
        metrics.context_chars,
        metrics.context_tokens_estimate,
    )
    qa_logger.debug("ANSWER: %s", answer)
    qa_logger.debug(
        "TIMING: generation_ms=%.0f | total_ms=%.0f",
        generation_time_ms,
        total_time_ms,
    )
    qa_logger.debug(separator)

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


def get_relevant_chunks_with_metrics(
    question: str, top_k: int = TOP_K_RESULTS
) -> RetrievalMetrics:
    """Retrieve relevant chunks from ChromaDB with timing metrics."""
    metrics = RetrievalMetrics()

    embed_start = time.perf_counter()
    model = get_embedding_model()
    question_embedding = model.encode([question])[0]
    metrics.embed_time_ms = (time.perf_counter() - embed_start) * 1000

    search_start = time.perf_counter()
    collection = get_chroma_collection()

    if collection.count() == 0:
        return metrics

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    metrics.search_time_ms = (time.perf_counter() - search_start) * 1000

    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i].get("source", "unknown")
        distance = results["distances"][0][i] if results["distances"] else 0.0
        metrics.chunks.append({
            "text": doc,
            "source": source,
            "distance": distance,
        })
        metrics.distances.append(distance)
        metrics.sources.append(source)

    return metrics


def get_relevant_chunks(question: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Retrieve relevant chunks from ChromaDB based on the question."""
    metrics = get_relevant_chunks_with_metrics(question, top_k)
    return metrics.chunks


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
    total_start = time.perf_counter()

    metrics = get_relevant_chunks_with_metrics(question)

    if not metrics.chunks:
        return "No documents found in the knowledge base. Please run the ingestion first."

    context = _build_context(metrics.chunks)
    metrics.context_chars = len(context)

    gen_start = time.perf_counter()
    answer = generate_answer(question, metrics.chunks)
    generation_time_ms = (time.perf_counter() - gen_start) * 1000
    total_time_ms = (time.perf_counter() - total_start) * 1000

    log_qa_interaction(question, answer, metrics, generation_time_ms, total_time_ms)

    return answer


def query_stream(question: str) -> Generator[str, None, None]:
    """Full RAG pipeline with streaming response."""
    total_start = time.perf_counter()

    metrics = get_relevant_chunks_with_metrics(question)

    if not metrics.chunks:
        yield "No documents found in the knowledge base. Please run the ingestion first."
        return

    context = _build_context(metrics.chunks)
    metrics.context_chars = len(context)

    gen_start = time.perf_counter()
    answer_parts: list[str] = []

    for token in generate_answer_stream(question, metrics.chunks):
        answer_parts.append(token)
        yield token

    generation_time_ms = (time.perf_counter() - gen_start) * 1000
    total_time_ms = (time.perf_counter() - total_start) * 1000
    full_answer = "".join(answer_parts)

    log_qa_interaction(question, full_answer, metrics, generation_time_ms, total_time_ms)


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
