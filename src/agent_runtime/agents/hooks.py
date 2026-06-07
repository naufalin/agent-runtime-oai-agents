"""RunHooks for automatic persistence of tool calls and responses."""

from dataclasses import dataclass

from agents import Agent, RunContextWrapper, RunHooks, Tool

from agent_runtime.db.session_repo import SessionRepo


@dataclass
class RunContext:
    """Context passed through the agent run for hooks to access persistence."""

    session_id: int
    repo: SessionRepo


class PersistenceHooks(RunHooks[RunContext]):
    """Automatically saves tool calls and responses to the database."""

    async def on_tool_start(
        self, context: RunContextWrapper[RunContext], agent: Agent, tool: Tool
    ) -> None:
        """Save a tool call message when a tool starts executing."""
        ctx = context.context
        await ctx.repo.add_message(
            ctx.session_id,
            role="tool",
            content=f"[calling {tool.name}...]",
            tool_name=tool.name,
        )

    async def on_tool_end(
        self, context: RunContextWrapper[RunContext], agent: Agent, tool: Tool, result: object
    ) -> None:
        """Save the tool response when a tool finishes executing."""
        ctx = context.context
        # Truncate very long tool results for readability
        content = str(result)
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        await ctx.repo.add_message(
            ctx.session_id,
            role="tool",
            content=content,
            tool_name=tool.name,
        )
