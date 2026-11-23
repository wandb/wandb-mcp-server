"""Shared HTTP test helpers for Agent Lens tools."""

import asyncio
import json
from contextlib import contextmanager
from typing import TypedDict
from unittest.mock import MagicMock, patch

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools import agent_lens as agent_lens_mod

BASE_URL = "https://agent-lens.example.com"
ENTITY = "acme"
PROJECT = "support-bot"
_Window = TypedDict("_Window", {"start_at": str, "end_at": str})
WINDOW: _Window = {"start_at": "2026-09-01T00:00:00Z", "end_at": "2026-09-08T00:00:00Z"}


def _sync_call(function, *args, **kwargs):
    return asyncio.run(function(*args, **kwargs))


@contextmanager
def _mocked(response, api_key="test-key", base_url=BASE_URL):
    """Point the module at a fake session, a fake key, and a fixed origin."""
    session = _FakeAsyncClient(response)

    def client_factory(timeout):
        session.timeout = timeout
        return session

    with (
        patch.object(agent_lens_mod, "_new_agent_lens_client", side_effect=client_factory),
        patch.object(WandBApiManager, "get_api_key", staticmethod(lambda: api_key)),
        patch.object(agent_lens_mod, "resolve_agent_lens_base_url", return_value=base_url),
        patch.object(agent_lens_mod, "track_tool_execution", _noop_tracker),
    ):
        yield session


@contextmanager
def _noop_tracker(*_args, **_kwargs):
    ctx = MagicMock()
    yield ctx


def _ok(payload):
    return _FakeResponse({"data": payload})


class _FakeResponse:
    def __init__(
        self,
        json_data,
        status_code=200,
        *,
        raw=None,
        headers=None,
        chunks=None,
        stream_error=None,
    ):
        self._json = json_data
        self.status_code = status_code
        self._raw = raw if raw is not None else json.dumps(json_data).encode("utf-8")
        self._chunks = chunks
        self._stream_error = stream_error
        self.headers = dict(headers or {})
        self.headers.setdefault(
            "Content-Length", str(sum(len(chunk) for chunk in chunks) if chunks else len(self._raw))
        )
        self.closed = False

    def json(self):
        if self._json is None:
            raise json.JSONDecodeError("no json", "", 0)
        return self._json

    async def aiter_raw(self, chunk_size=None):
        if self._stream_error is not None:
            raise self._stream_error
        if self._chunks is not None:
            for chunk in self._chunks:
                yield chunk
            return
        chunk_size = chunk_size or len(self._raw) or 1
        for offset in range(0, len(self._raw), chunk_size):
            yield self._raw[offset : offset + chunk_size]

    async def aclose(self):
        self.closed = True


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    async def __aexit__(self, *_args):
        await self._response.aclose()


class _FakeAsyncClient:
    """Records requests and returns a canned response."""

    def __init__(self, response):
        self._response = response
        self.calls = []
        self.timeout = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def stream(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if isinstance(self._response, list) and self._response:
            next_resp = self._response.pop(0) if len(self._response) > 1 else self._response[0]
            return _FakeStreamContext(next_resp)
        return _FakeStreamContext(self._response)

    @property
    def last(self):
        return self.calls[-1]
