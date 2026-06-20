"""Bearer token authentication for protected API routes."""

import secrets

from fastapi import Header, HTTPException, Query

from agent_runtime.config import Settings


def _auth_error() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_bearer_token(authorization: str | None = Header(default=None)) -> str:
    """Require a shared bearer token for private API routes."""
    return _validate_token(authorization=authorization)


async def require_docs_token(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> str:
    """Require auth for docs routes, allowing browser-friendly ?token=... access."""
    return _validate_token(authorization=authorization, query_token=token)


def _validate_token(
    authorization: str | None = None,
    query_token: str | None = None,
) -> str:
    expected_token = Settings().agent_runtime_bearer_token.strip()
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Bearer token is not configured",
        )

    if query_token and secrets.compare_digest(query_token, expected_token):
        return query_token

    if not authorization:
        raise _auth_error()

    scheme, separator, bearer_token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not bearer_token:
        raise _auth_error()

    if not secrets.compare_digest(bearer_token, expected_token):
        raise _auth_error()

    return bearer_token
