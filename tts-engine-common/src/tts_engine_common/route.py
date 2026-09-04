"""FastAPI route factory for the GET /capabilities endpoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .models import Capabilities


def capabilities_endpoint(doc: Capabilities):
    """Return an async handler serving the capabilities document as JSON.

    The document is serialized once at build time — it is static for the
    lifetime of the server, so each request is a dict echo rather than a
    re-serialization.

    No Cache-Control header is sent on purpose (design question Q5): the
    client app re-interrogates /capabilities on every initial connection and
    must see server changes immediately, not after a cache expiry.
    """
    payload = doc.model_dump(mode="json")

    async def _capabilities() -> JSONResponse:
        return JSONResponse(content=payload)

    return _capabilities


def add_capabilities_route(app: FastAPI, doc: Capabilities, path: str = "/capabilities") -> None:
    """Convenience wrapper: mount /capabilities on an existing FastAPI app."""
    app.add_api_route(path, capabilities_endpoint(doc), methods=["GET"])
