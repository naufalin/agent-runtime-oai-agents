"""Unit tests for CLI argument parsing and commands."""

from unittest.mock import AsyncMock, patch, MagicMock
import pytest


class TestCliMain:
    def test_no_args_starts_chat_loop(self):
        with (
            patch("agent_runtime.cli.chat_loop", new_callable=MagicMock) as mock_fn,
            patch("agent_runtime.cli.asyncio.run") as mock_run,
            patch("agent_runtime.cli.sys.argv", ["agent-runtime-cli"]),
        ):
            from agent_runtime.cli import cli_main
            cli_main()
            mock_run.assert_called_once()

    def test_list_calls_list_sessions(self):
        with (
            patch("agent_runtime.cli.list_sessions", new_callable=MagicMock) as mock_fn,
            patch("agent_runtime.cli.asyncio.run") as mock_run,
            patch("agent_runtime.cli.sys.argv", ["agent-runtime-cli", "list"]),
        ):
            from agent_runtime.cli import cli_main
            cli_main()
            mock_run.assert_called_once()

    def test_resume_with_id(self):
        with (
            patch("agent_runtime.cli.chat_loop", new_callable=MagicMock) as mock_fn,
            patch("agent_runtime.cli.asyncio.run") as mock_run,
            patch("agent_runtime.cli.sys.argv", ["agent-runtime-cli", "resume", "abc123"]),
        ):
            from agent_runtime.cli import cli_main
            cli_main()
            mock_run.assert_called_once()

    def test_resume_without_id_exits_1(self):
        with (
            patch("agent_runtime.cli.sys.argv", ["agent-runtime-cli", "resume"]),
            pytest.raises(SystemExit, match="1"),
        ):
            from agent_runtime.cli import cli_main
            cli_main()

    def test_prompts_command(self):
        with (
            patch("agent_runtime.cli._list_prompts", new_callable=MagicMock) as mock_fn,
            patch("agent_runtime.cli.asyncio.run") as mock_run,
            patch("agent_runtime.cli.sys.argv", ["agent-runtime-cli", "prompts"]),
        ):
            from agent_runtime.cli import cli_main
            cli_main()
            mock_run.assert_called_once()

    def test_create_prompt_command(self):
        with (
            patch("agent_runtime.cli.create_prompt", new_callable=MagicMock) as mock_fn,
            patch("agent_runtime.cli.asyncio.run") as mock_run,
            patch("agent_runtime.cli.sys.argv", ["agent-runtime-cli", "create-prompt", "pirate", "Arr!"]),
        ):
            from agent_runtime.cli import cli_main
            cli_main()
            mock_run.assert_called_once()

    def test_create_prompt_missing_args_exits_1(self):
        with (
            patch("agent_runtime.cli.sys.argv", ["agent-runtime-cli", "create-prompt"]),
            pytest.raises(SystemExit, match="1"),
        ):
            from agent_runtime.cli import cli_main
            cli_main()

    def test_unknown_command_exits_1(self):
        with (
            patch("agent_runtime.cli.sys.argv", ["agent-runtime-cli", "foobar"]),
            pytest.raises(SystemExit, match="1"),
        ):
            from agent_runtime.cli import cli_main
            cli_main()


@pytest.mark.asyncio
async def test_list_sessions_prints_table():
    mock_db = AsyncMock()
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()

    from agent_runtime.db.models import SystemPrompt
    from types import SimpleNamespace

    mock_repo.list_sessions.return_value = [
        SimpleNamespace(id=1, title="Session A", updated_at=None),
    ]
    mock_repo.get_latest_system_message.return_value = SimpleNamespace(
        id=1, system_prompt_id=1,
    )
    mock_prompt_repo.get_by_id.return_value = SystemPrompt(id=1, name="default", content="Helpful.")

    with (
        patch("agent_runtime.cli.get_db", return_value=mock_db),
        patch("agent_runtime.cli.SessionRepo", return_value=mock_repo),
        patch("agent_runtime.cli.SystemPromptRepo", return_value=mock_prompt_repo),
    ):
        from agent_runtime.cli import list_sessions
        await list_sessions()

    mock_repo.list_sessions.assert_called_once_with(limit=10)
