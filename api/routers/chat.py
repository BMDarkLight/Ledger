"""`POST /v1/chat/completions`: OpenAI-compatible surface.

Existing clients point at Ledger unchanged. Receipts survive the translation:
the tags stay inline in the message content, and the structured receipts ride
along in a non-standard `ledger` field that compliant clients ignore.

Streaming does not stream the model's tokens. An answer is checked against its
receipts as a whole, so nothing is sent until the check has run; the checked
answer is then streamed sentence by sentence, each sentence with its tags. That
costs time to first token, and it is the price of never emitting a claim the
gate would have flagged.
"""

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.deps import SettingsDep
from api.routers.ask import _guarded
from api.schemas import AskRequest, ReceiptedAskResponse
from api.services.receipts import SENTENCE_BOUNDARY

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "ledger"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


def _last_user_question(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    raise HTTPException(status_code=400, detail="no user message in the conversation")


def _ledger_block(result: ReceiptedAskResponse) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "route": result.route.model_dump(mode="json"),
        "receipts": [r.model_dump(mode="json") for r in result.receipts],
        "citation_coverage": result.citation_coverage,
    }


def _sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def _pieces(text: str) -> list[str]:
    """Sentence-sized slices of `text` that join back to exactly `text`.

    Cut at the same boundaries the receipts check uses, so a claim and its tags
    always arrive in the same chunk.
    """
    pieces, start = [], 0
    for boundary in SENTENCE_BOUNDARY.finditer(text):
        pieces.append(text[start : boundary.end()])
        start = boundary.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def _stream(result: ReceiptedAskResponse, model: str) -> Iterator[str]:
    """The checked answer as OpenAI `chat.completion.chunk` events.

    The receipts ride on the final chunk, after the last sentence, so a client
    that reads them has seen everything they account for.
    """
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }

    def chunk(delta: dict[str, str], finish: str | None = None) -> dict[str, Any]:
        return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield _sse(chunk({"role": "assistant", "content": ""}))
    for piece in _pieces(result.answer):
        yield _sse(chunk({"content": piece}))
    yield _sse({**chunk({}, finish="stop"), "ledger": _ledger_block(result)})
    yield _sse("[DONE]")


@router.post("/chat/completions", response_model=None)
def chat_completions(
    request: ChatCompletionRequest, settings: SettingsDep
) -> dict[str, Any] | StreamingResponse:
    question = _last_user_question(request.messages)

    # The pipeline runs to completion before anything is sent, streaming or
    # not, so a failure is still an HTTP error rather than a broken stream.
    result = _guarded(AskRequest(question=question), settings)

    if request.stream:
        return StreamingResponse(_stream(result, request.model), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "ledger": _ledger_block(result),
    }
