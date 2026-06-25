"""Root conftest — shared across all test layers."""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply unit/integration markers based on test path."""
    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
