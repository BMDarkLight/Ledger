"""Ledger's HTTP surface."""

from fastapi import FastAPI

from api import __version__
from api.routers import ask, chat, documents, route, tools
from api.routers import eval as eval_router
from api.schemas import HealthResponse

app = FastAPI(
    title="Ledger",
    description="Nothing enters the answer without a receipt.",
    version=__version__,
)

app.include_router(route.router)
app.include_router(ask.router)
app.include_router(documents.router)
app.include_router(tools.router)
app.include_router(eval_router.router)
app.include_router(chat.router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(version=__version__)
