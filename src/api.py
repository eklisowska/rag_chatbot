import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.clients import get_chroma_collection
from src.rag import get_all_chunks, query, query_stream

logger = logging.getLogger(__name__)

API_MODEL_NAME = "rag-chatbot"
API_OWNER = "local"
CHARS_PER_TOKEN_ESTIMATE = 4

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

executor = ThreadPoolExecutor(max_workers=4)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = API_MODEL_NAME
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models (OpenAI-compatible)."""
    return ModelList(
        data=[
            ModelInfo(
                id=API_MODEL_NAME,
                created=int(time.time()),
                owned_by=API_OWNER,
            )
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    """Get model info (OpenAI-compatible)."""
    return ModelInfo(
        id=model_id,
        created=int(time.time()),
        owned_by=API_OWNER,
    )


def format_sse_message(data: dict) -> str:
    """Format a message for Server-Sent Events."""
    return f"data: {json.dumps(data)}\n\n"


def _generate_response_id() -> str:
    """Generate a unique response ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def _extract_user_question(messages: list[Message]) -> str:
    """Extract the last user message from the conversation."""
    user_messages = [m for m in messages if m.role == "user"]
    return user_messages[-1].content if user_messages else "Hello"


def _create_stream_chunk(
    response_id: str,
    created: int,
    content: str | None = None,
    finish_reason: str | None = None,
) -> dict:
    """Create a streaming response chunk."""
    delta = {"content": content} if content is not None else {}
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": API_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _estimate_tokens(text: str) -> int:
    """Estimate token count based on character length."""
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Extracts the last user message and runs the RAG pipeline.
    """
    question = _extract_user_question(request.messages)
    logger.info("Received question: %s", question[:100])

    if request.stream:
        async def generate_stream() -> AsyncGenerator[str, None]:
            response_id = _generate_response_id()
            created = int(time.time())
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def stream_to_queue():
                try:
                    for token in query_stream(question):
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            executor.submit(stream_to_queue)

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield format_sse_message(
                    _create_stream_chunk(response_id, created, content=token)
                )

            yield format_sse_message(
                _create_stream_chunk(response_id, created, finish_reason="stop")
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(executor, query, question)

    prompt_tokens = _estimate_tokens(question)
    completion_tokens = _estimate_tokens(answer)

    return ChatCompletionResponse(
        id=_generate_response_id(),
        created=int(time.time()),
        model=API_MODEL_NAME,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.get("/v1/chunks")
async def list_chunks(
    limit: int | None = Query(None, ge=1, description="Maximum chunks to return"),
    offset: int = Query(0, ge=0, description="Number of chunks to skip"),
):
    """
    List all chunks stored in the knowledge base.
    
    Args:
        limit: Maximum number of chunks to return (None = all)
        offset: Number of chunks to skip (for pagination)
    """
    return get_all_chunks(limit=limit, offset=offset)


@app.get("/health")
async def health_check():
    """Health check endpoint that verifies ChromaDB connectivity."""
    try:
        collection = get_chroma_collection()
        doc_count = collection.count()
        return {
            "status": "healthy",
            "chromadb": "connected",
            "documents": doc_count,
        }
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "chromadb": "disconnected",
                "error": str(e),
            },
        )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "chunks": "/v1/chunks",
            "health": "/health",
        },
    }
