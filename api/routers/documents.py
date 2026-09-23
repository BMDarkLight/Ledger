"""`POST /v1/documents`: ingestion into the vector store."""

from fastapi import APIRouter, HTTPException

from api.deps import SettingsDep
from api.schemas import DocumentRequest, DocumentResponse
from api.services import retrieval

router = APIRouter(prefix="/v1", tags=["documents"])


@router.post("/documents", response_model=DocumentResponse)
def ingest(request: DocumentRequest, settings: SettingsDep) -> DocumentResponse:
    try:
        chunks = retrieval.index_document(request.doc_id, request.text, request.metadata, settings)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return DocumentResponse(doc_id=request.doc_id, chunks_indexed=chunks)
