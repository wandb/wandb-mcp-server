"""Tests for the temporary W&B Public API relogin shim."""

from types import SimpleNamespace

from wandb_mcp_server.wandb_public_api_shim import (
    APPLIED_ATTR,
    DISABLE_ENV_VAR,
    ORIGINAL_ATTR,
    USER_AGENT_PREFIX,
    apply_public_api_relogin_shim,
)


def _fake_public_api_module():
    calls = []

    def service_api(*args, settings=None, **kwargs):
        calls.append((args, settings, kwargs))
        return {"settings": settings}

    return SimpleNamespace(ServiceApi=service_api, calls=calls)


def test_public_api_shim_injects_headers_when_missing():
    public_api = _fake_public_api_module()
    settings = SimpleNamespace(x_extra_http_headers=None)

    assert apply_public_api_relogin_shim(public_api) is True
    result = public_api.ServiceApi(settings=settings)

    assert result == {"settings": settings}
    assert settings.x_extra_http_headers["Use-Admin-Privileges"] == "true"
    assert settings.x_extra_http_headers["User-Agent"].startswith(USER_AGENT_PREFIX)
    assert getattr(public_api, APPLIED_ATTR) is True
    assert getattr(public_api, ORIGINAL_ATTR) is not public_api.ServiceApi


def test_public_api_shim_preserves_existing_user_agent():
    public_api = _fake_public_api_module()
    settings = SimpleNamespace(
        x_extra_http_headers={
            "Use-Admin-Privileges": "already-set",
            "User-Agent": "W&B Public Client fixed-sdk",
        }
    )

    apply_public_api_relogin_shim(public_api)
    public_api.ServiceApi(settings=settings)

    assert settings.x_extra_http_headers["Use-Admin-Privileges"] == "already-set"
    assert settings.x_extra_http_headers["User-Agent"] == "W&B Public Client fixed-sdk"


def test_public_api_shim_is_idempotent():
    public_api = _fake_public_api_module()

    assert apply_public_api_relogin_shim(public_api) is True
    wrapped = public_api.ServiceApi
    assert apply_public_api_relogin_shim(public_api) is False

    assert public_api.ServiceApi is wrapped


def test_public_api_shim_can_be_disabled(monkeypatch):
    public_api = _fake_public_api_module()
    settings = SimpleNamespace(x_extra_http_headers=None)
    monkeypatch.setenv(DISABLE_ENV_VAR, "1")

    assert apply_public_api_relogin_shim(public_api) is False
    public_api.ServiceApi(settings=settings)

    assert settings.x_extra_http_headers is None
    assert not hasattr(public_api, APPLIED_ATTR)
