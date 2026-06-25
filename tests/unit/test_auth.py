"""Unit tests for bearer token authentication."""

import pytest

from agent_runtime.api.auth import _validate_token


@pytest.fixture(autouse=True)
def _set_token(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_BEARER_TOKEN", "test-secret")


class TestBearerHeader:
    def test_valid_bearer_token_passes(self):
        result = _validate_token(authorization="Bearer test-secret")
        assert result == "test-secret"

    def test_wrong_token_returns_401(self):
        with pytest.raises(Exception, match="401"):
            _validate_token(authorization="Bearer wrong-token")

    def test_missing_header_returns_401(self):
        with pytest.raises(Exception, match="401"):
            _validate_token(authorization=None)

    def test_empty_header_returns_401(self):
        with pytest.raises(Exception, match="401"):
            _validate_token(authorization="")

    def test_malformed_scheme_basic_returns_401(self):
        with pytest.raises(Exception, match="401"):
            _validate_token(authorization="Basic dXNlcjpwYXNz")

    def test_malformed_scheme_token_returns_401(self):
        with pytest.raises(Exception, match="401"):
            _validate_token(authorization="Token test-secret")

    def test_bearer_without_value_returns_401(self):
        with pytest.raises(Exception, match="401"):
            _validate_token(authorization="Bearer ")

    def test_bearer_lowercase_passes(self):
        """Scheme matching is case-insensitive."""
        result = _validate_token(authorization="bearer test-secret")
        assert result == "test-secret"

    def test_bearer_mixed_case_passes(self):
        result = _validate_token(authorization="BEARER test-secret")
        assert result == "test-secret"


class TestQueryToken:
    def test_valid_query_token_passes(self):
        result = _validate_token(authorization=None, query_token="test-secret")
        assert result == "test-secret"

    def test_wrong_query_token_returns_401(self):
        with pytest.raises(Exception, match="401"):
            _validate_token(authorization=None, query_token="wrong")

    def test_query_token_takes_precedence_over_header(self):
        """When both are provided, query token is checked first."""
        result = _validate_token(
            authorization="Bearer wrong", query_token="test-secret"
        )
        assert result == "test-secret"


class TestEmptyConfiguredToken:
    def test_empty_configured_token_returns_503(self, monkeypatch):
        monkeypatch.setenv("AGENT_RUNTIME_BEARER_TOKEN", "")
        with pytest.raises(Exception, match="503"):
            _validate_token(authorization="Bearer anything")

    def test_whitespace_only_configured_token_returns_503(self, monkeypatch):
        monkeypatch.setenv("AGENT_RUNTIME_BEARER_TOKEN", "   ")
        with pytest.raises(Exception, match="503"):
            _validate_token(authorization="Bearer anything")
