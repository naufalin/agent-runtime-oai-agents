"""Bearer token authentication for protected API routes."""

import secrets

from fastapi import Header, HTTPException

from agent_runtime.config import Settings


def _auth_error() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_bearer_token(authorization: str | None = Header(default=None)) -> str:
    """Require a shared bearer token for private API routes."""
    expected_token = Settings().agent_runtime_bearer_token.strip()
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Bearer token is not configured",
        )

    if not authorization:
        raise _auth_error()

    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        raise _auth_error()

    if not secrets.compare_digest(token, expected_token):
        raise _auth_error()

    return token
