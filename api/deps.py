"""Shared FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends

from api.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]
