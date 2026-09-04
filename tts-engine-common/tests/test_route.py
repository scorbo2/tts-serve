"""Tests for the /capabilities route factory."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from tts_engine_common import build_capabilities, capabilities_endpoint


def _app() -> FastAPI:
    class Request(BaseModel):
        text: str = Field(..., min_length=1)
        language: Literal["en", "de"] | None = None
        seed: int | None = Field(None, ge=1, le=1000)

    doc = build_capabilities(
        Request,
        engine="test-engine",
        model="test-model",
        device="cpu",
        sample_rate=24000,
        watermarked=False,
    )
    app = FastAPI()
    app.add_api_route("/capabilities", capabilities_endpoint(doc), methods=["GET"])
    return app


def test_route_capabilities_returnsDocumentAsJson() -> None:
    # GIVEN a FastAPI app with the capabilities route mounted:
    app = _app()

    # WHEN the endpoint is called:
    # (TestClient is used without a context manager on purpose: no lifespan,
    # so this works on a dev box with no model loaded.)
    with TestClient(app) as client:
        response = client.get("/capabilities")

    # THEN it serves the full document as JSON:
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["engine"] == "test-engine"
    assert body["sample_rate"] == 24000
    assert {p["name"] for p in body["parameters"]} == {"text", "language", "seed"}


def test_route_capabilities_bodyMatchesModelDump() -> None:
    # GIVEN a FastAPI app with the capabilities route mounted:
    app = _app()

    # WHEN the endpoint is called:
    with TestClient(app) as client:
        body = client.get("/capabilities").json()

    # THEN the payload equals the document's model_dump(mode="json") exactly:
    class Request(BaseModel):
        text: str = Field(..., min_length=1)
        language: Literal["en", "de"] | None = None
        seed: int | None = Field(None, ge=1, le=1000)

    expected = build_capabilities(
        Request,
        engine="test-engine",
        model="test-model",
        device="cpu",
        sample_rate=24000,
        watermarked=False,
    ).model_dump(mode="json")
    assert body == expected


def test_route_capabilities_sendsNoCacheControl() -> None:
    # GIVEN a FastAPI app with the capabilities route mounted:
    app = _app()

    # WHEN the endpoint is called:
    with TestClient(app) as client:
        response = client.get("/capabilities")

    # THEN no Cache-Control header is sent (design question Q5: the client app
    # re-interrogates on every initial connection and must see updates):
    assert "cache-control" not in {k.lower() for k in response.headers}
