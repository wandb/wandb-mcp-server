"""The installed tokenizer must not need a network or writable cache."""

import hashlib
import socket

import pytest
import tiktoken

from wandb_mcp_server.trace_utils import _get_tiktoken_encoding


def test_tokenizer_works_without_registry_fetch_or_cache(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("tokenizer attempted network access")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "absent-cache"))
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(tiktoken, "get_encoding", no_network)
    _get_tiktoken_encoding.cache_clear()
    encoding = _get_tiktoken_encoding()
    assert encoding.encode("hello world") == [15339, 1917]
    assert encoding.encode("你好") == [57668, 53901]
    assert not (tmp_path / "absent-cache").exists()


def test_packaged_vocabulary_is_the_pinned_asset():
    from importlib.resources import files
    from wandb_mcp_server.tokenizer import TOKENIZER_SHA256

    data = files("wandb_mcp_server").joinpath("data", "cl100k_base.tiktoken").read_bytes()
    assert hashlib.sha256(data).hexdigest() == TOKENIZER_SHA256


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_corrupt_tokenizer_fails_registration(monkeypatch, tmp_path, missing):
    from wandb_mcp_server import tokenizer
    from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
    from wandb_mcp_server.server import register_tools

    if not missing:
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "cl100k_base.tiktoken").write_bytes(b"corrupt")
    monkeypatch.setattr(tokenizer, "files", lambda package: tmp_path)
    tokenizer.load_tokenizer.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="tokenizer"):
            register_tools(InstrumentedFastMCP("offline-audit"))
    finally:
        tokenizer.load_tokenizer.cache_clear()


def test_all_runtime_token_counters_share_offline_encoding(monkeypatch):
    from wandb_mcp_server.trace_utils import count_tokens, count_tokens_conservative
    from wandb_mcp_server.weave_api.processors import TraceProcessor

    def no_registry(*args, **kwargs):
        raise AssertionError("no runtime registry access")

    monkeypatch.setattr(tiktoken, "get_encoding", no_registry)
    text = "你好" * 100
    assert count_tokens(text) == count_tokens_conservative(text) == TraceProcessor.count_tokens(text) == 200


def test_reserved_token_spelling_is_counted_as_ordinary_text():
    from wandb_mcp_server.trace_utils import count_tokens_conservative

    text = "<|endoftext|>"
    assert count_tokens_conservative(text) == len(_get_tiktoken_encoding().encode_ordinary(text))
