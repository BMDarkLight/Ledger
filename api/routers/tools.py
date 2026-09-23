"""`GET /v1/tools`: what Ledger can call, and what is actually configured."""

from fastapi import APIRouter

from api.deps import SettingsDep
from api.schemas import ToolSpec
from api.services import tools as tools_service

router = APIRouter(prefix="/v1", tags=["tools"])


@router.get("/tools", response_model=list[ToolSpec])
def list_tools(settings: SettingsDep) -> list[ToolSpec]:
    return tools_service.list_tools(settings)
