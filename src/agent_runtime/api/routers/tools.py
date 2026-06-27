"""Tool introspection endpoint.

Exposes the names + descriptions of tools available in the runtime. Clients
use this to render a tool picker without hardcoding names.
"""

from fastapi import APIRouter

from agent_runtime.agents.runtime import TOOL_REGISTRY, available_tool_names
from agent_runtime.api.schemas import ToolListOut, ToolOut

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("", response_model=ToolListOut)
async def list_tools() -> ToolListOut:
    items = [
        ToolOut(name=name, description=TOOL_REGISTRY[name].description)
        for name in available_tool_names()
    ]
    return ToolListOut(tools=items, total=len(items))
