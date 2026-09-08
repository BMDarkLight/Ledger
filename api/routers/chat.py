"""`POST /v1/chat/completions` — OpenAI-compatible surface.

Existing clients point at Ledger unchanged. Receipts survive the translation:
the tags stay inline in the message content, and the structured receipts ride
along in a non-standard `ledger` field that compliant clients ignore.
"""

import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import SettingsDep
from api.routers.ask import _guarded
from api.schemas import AskRequest

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


@router.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest, settings: SettingsDep) -> dict[str, Any]:
    question = _last_user_question(request.messages)
    if request.stream:
        raise HTTPException(status_code=501, detail="Phase 5: SSE streaming.")

    result = _guarded(AskRequest(question=question), settings)

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
        "ledger": {
            "status": result.status.value,
            "route": result.route.model_dump(mode="json"),
            "receipts": [r.model_dump(mode="json") for r in result.receipts],
            "citation_coverage": result.citation_coverage,
        },
    }
