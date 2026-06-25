"""Unit test fixtures — no DB, no network."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
