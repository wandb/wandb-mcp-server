"""Unit tests for OAuth 2.1 token validation, exchange, and dual-mode auth.

All tests are mock-based -- no network calls, no real JWKS endpoints.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from wandb_mcp_server.oauth import (
    OAuthConfig,
    get_oauth_config,
    is_jwt,
    reset_oauth_config,
    validate_oauth_token,
)


# ---------------------------------------------------------------------------
# Helpers: generate RSA key pair for test JWTs
# ---------------------------------------------------------------------------


def _generate_rsa_keypair():
    """Generate an RSA private key and return (private_key, public_key)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _make_jwks_dict(public_key, kid="test-key-1"):
    """Build a JWKS dict from an RSA public key."""
    from jwt.algorithms import RSAAlgorithm

    jwk = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    return {"keys": [jwk]}


def _make_multi_key_jwks(public_key_1, public_key_2, kid_1="key-1", kid_2="key-2"):
    """Build a JWKS dict with two keys."""
    from jwt.algorithms import RSAAlgorithm

    jwk1 = json.loads(RSAAlgorithm.to_jwk(public_key_1))
    jwk1["kid"] = kid_1
    jwk1["use"] = "sig"
    jwk1["alg"] = "RS256"

    jwk2 = json.loads(RSAAlgorithm.to_jwk(public_key_2))
    jwk2["kid"] = kid_2
    jwk2["use"] = "sig"
    jwk2["alg"] = "RS256"

    return {"keys": [jwk1, jwk2]}


def _encode_jwt(claims: dict, private_key, kid="test-key-1") -> str:
    """Encode a JWT signed with the test RSA key."""
    return pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def _encode_jwt_no_kid(claims: dict, private_key) -> str:
    """Encode a JWT without a kid header."""
    return pyjwt.encode(claims, private_key, algorithm="RS256")


# ---------------------------------------------------------------------------
# is_jwt
# ---------------------------------------------------------------------------


class TestIsJwt:
    def test_valid_jwt_shape(self):
        assert is_jwt("eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature") is True

    def test_api_key_is_not_jwt(self):
        assert is_jwt("361c051b8ff10b27672677cd6735c0ce3c32fda0") is False

    def test_two_parts_is_not_jwt(self):
        assert is_jwt("header.payload") is False

    def test_empty_segment_is_not_jwt(self):
        assert is_jwt("header..signature") is False

    def test_empty_string(self):
        assert is_jwt("") is False


# ---------------------------------------------------------------------------
# OAuthConfig
# ---------------------------------------------------------------------------


class TestOAuthConfig:
    def setup_method(self):
        reset_oauth_config()

    def test_from_env_with_all_vars(self):
        with patch.dict(
            "os.environ",
            {
                "MCP_OAUTH_ISSUER": "https://auth.example.com",
                "MCP_OAUTH_AUDIENCE": "https://mcp.example.com",
                "MCP_OAUTH_REQUIRED_SCOPES": "mcp:read,mcp:write",
            },
        ):
            config = OAuthConfig.from_env()
            assert config.issuer == "https://auth.example.com"
            assert config.audience == "https://mcp.example.com"
            assert config.required_scopes == ["mcp:read", "mcp:write"]

    def test_from_env_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            config = OAuthConfig.from_env()
            assert config.issuer == ""
            assert config.audience == ""
            assert config.required_scopes == []

    def test_from_env_empty_scopes(self):
        with patch.dict("os.environ", {"MCP_OAUTH_REQUIRED_SCOPES": ""}):
            config = OAuthConfig.from_env()
            assert config.required_scopes == []

    def test_get_oauth_config_singleton(self):
        reset_oauth_config()
        with patch.dict("os.environ", {"MCP_OAUTH_ISSUER": "https://test.com"}):
            c1 = get_oauth_config()
            c2 = get_oauth_config()
            assert c1 is c2
            assert c1.issuer == "https://test.com"
        reset_oauth_config()


# ---------------------------------------------------------------------------
# validate_oauth_token
# ---------------------------------------------------------------------------


class TestValidateOauthToken:
    @pytest.fixture(autouse=True)
    def _setup_keys(self):
        self.private_key, self.public_key = _generate_rsa_keypair()
        self.jwks = _make_jwks_dict(self.public_key)
        self.issuer = "https://auth.example.com"
        self.config = OAuthConfig(
            issuer=self.issuer,
            audience="https://mcp.example.com",
            required_scopes=[],
        )
        import wandb_mcp_server.oauth as oauth_mod

        oauth_mod._jwks_clients.clear()

    def _mock_jwks(self, jwks_data=None):
        """Patch httpx.get to return OIDC discovery + JWKS."""
        data = jwks_data if jwks_data is not None else self.jwks

        discovery_resp = MagicMock()
        discovery_resp.json.return_value = {"jwks_uri": f"{self.issuer}/.well-known/jwks.json"}
        discovery_resp.raise_for_status = MagicMock()

        jwks_resp = MagicMock()
        jwks_resp.json.return_value = data
        jwks_resp.raise_for_status = MagicMock()

        def side_effect(url, **kwargs):
            if "openid-configuration" in url:
                return discovery_resp
            return jwks_resp

        return patch("wandb_mcp_server.oauth.httpx.get", side_effect=side_effect)

    def test_valid_token(self):
        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
            },
            self.private_key,
        )
        with self._mock_jwks():
            claims = validate_oauth_token(token, self.config)
            assert claims["sub"] == "user-123"

    def test_expired_token_rejected(self):
        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) - 100,
                "iat": int(time.time()) - 200,
            },
            self.private_key,
        )
        with self._mock_jwks():
            with pytest.raises(pyjwt.ExpiredSignatureError):
                validate_oauth_token(token, self.config)

    def test_wrong_audience_rejected(self):
        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://wrong.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
            },
            self.private_key,
        )
        with self._mock_jwks():
            with pytest.raises(pyjwt.InvalidAudienceError):
                validate_oauth_token(token, self.config)

    def test_wrong_issuer_rejected(self):
        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": "https://evil.com",
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
            },
            self.private_key,
        )
        with self._mock_jwks():
            with pytest.raises(pyjwt.InvalidIssuerError):
                validate_oauth_token(token, self.config)

    def test_missing_required_scopes_rejected(self):
        self.config.required_scopes = ["mcp:read", "mcp:write"]
        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "scope": "mcp:read",
            },
            self.private_key,
        )
        with self._mock_jwks():
            with pytest.raises(pyjwt.InvalidTokenError, match="missing required scopes"):
                validate_oauth_token(token, self.config)

    def test_all_required_scopes_present(self):
        self.config.required_scopes = ["mcp:read", "mcp:write"]
        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "scope": "mcp:read mcp:write openid",
            },
            self.private_key,
        )
        with self._mock_jwks():
            claims = validate_oauth_token(token, self.config)
            assert claims["sub"] == "user-123"

    def test_scopes_as_list_accepted(self):
        """Gorilla and some providers return scopes as a JSON array (scp claim)."""
        self.config.required_scopes = ["mcp:read"]
        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "scp": ["mcp:read", "mcp:write"],
            },
            self.private_key,
        )
        with self._mock_jwks():
            claims = validate_oauth_token(token, self.config)
            assert claims["sub"] == "user-123"

    def test_no_issuer_configured_raises(self):
        config = OAuthConfig(issuer="", audience="", required_scopes=[])
        with pytest.raises(pyjwt.InvalidTokenError, match="MCP_OAUTH_ISSUER not configured"):
            validate_oauth_token("eyJ.eyJ.sig", config)

    def test_no_audience_skips_audience_check(self):
        config = OAuthConfig(issuer=self.issuer, audience="", required_scopes=[])
        token = _encode_jwt(
            {"sub": "user-123", "iss": self.issuer, "exp": int(time.time()) + 3600, "iat": int(time.time())},
            self.private_key,
        )
        with self._mock_jwks():
            claims = validate_oauth_token(token, config)
            assert claims["sub"] == "user-123"

    def test_kid_mismatch_rejected(self):
        """Token signed with kid=X but JWKS only has kid=Y."""
        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
            },
            self.private_key,
            kid="wrong-key-id",
        )
        with self._mock_jwks():
            with pytest.raises(pyjwt.InvalidTokenError, match="No JWKS key matching kid"):
                validate_oauth_token(token, self.config)

    def test_no_kid_multi_key_jwks_rejected(self):
        """JWT without kid header and JWKS has multiple keys -- must reject."""
        _, pub2 = _generate_rsa_keypair()
        multi_jwks = _make_multi_key_jwks(self.public_key, pub2)
        token = _encode_jwt_no_kid(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
            },
            self.private_key,
        )
        with self._mock_jwks(jwks_data=multi_jwks):
            with pytest.raises(pyjwt.InvalidTokenError, match="cannot select"):
                validate_oauth_token(token, self.config)

    def test_no_kid_single_key_jwks_accepted(self):
        """JWT without kid header but JWKS has exactly one key -- accept."""
        token = _encode_jwt_no_kid(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
            },
            self.private_key,
        )
        with self._mock_jwks():
            claims = validate_oauth_token(token, self.config)
            assert claims["sub"] == "user-123"


# ---------------------------------------------------------------------------
# Token exchange (oauth_exchange.py)
# ---------------------------------------------------------------------------


class TestTokenExchange:
    def setup_method(self):
        from wandb_mcp_server.oauth_exchange import clear_exchange_cache

        clear_exchange_cache()

    def test_exchange_success(self):
        from wandb_mcp_server.oauth_exchange import exchange_jwt_for_wb_token

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "wb_at_test123.sig456", "expires_in": 3600}

        with patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=mock_resp):
            result = exchange_jwt_for_wb_token("fake.jwt.token", "https://api.wandb.ai")
            assert result.token == "wb_at_test123.sig456"
            assert not result.is_expired

    def test_exchange_failure_raises(self):
        from wandb_mcp_server.oauth_exchange import OAuthExchangeError, exchange_jwt_for_wb_token

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=mock_resp):
            with pytest.raises(OAuthExchangeError, match="401"):
                exchange_jwt_for_wb_token("fake.jwt.token", "https://api.wandb.ai")

    def test_exchange_network_error_raises(self):
        from wandb_mcp_server.oauth_exchange import OAuthExchangeError, exchange_jwt_for_wb_token

        with patch("wandb_mcp_server.oauth_exchange.httpx.post", side_effect=Exception("connection refused")):
            with pytest.raises(OAuthExchangeError, match="connection refused"):
                exchange_jwt_for_wb_token("fake.jwt.token", "https://api.wandb.ai")

    def test_exchange_cache_reuse(self):
        from wandb_mcp_server.oauth_exchange import exchange_jwt_for_wb_token

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "wb_at_cached.sig", "expires_in": 3600}

        with patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=mock_resp) as mock_post:
            r1 = exchange_jwt_for_wb_token("same.jwt.token", "https://api.wandb.ai")
            r2 = exchange_jwt_for_wb_token("same.jwt.token", "https://api.wandb.ai")

            assert r1.token == r2.token
            assert mock_post.call_count == 1

    def test_exchange_cache_miss_different_jwt(self):
        from wandb_mcp_server.oauth_exchange import exchange_jwt_for_wb_token

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "wb_at_x.sig", "expires_in": 3600}

        with patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=mock_resp) as mock_post:
            exchange_jwt_for_wb_token("jwt.one.token", "https://api.wandb.ai")
            exchange_jwt_for_wb_token("jwt.two.token", "https://api.wandb.ai")
            assert mock_post.call_count == 2

    def test_exchange_malformed_response_raises(self):
        from wandb_mcp_server.oauth_exchange import OAuthExchangeError, exchange_jwt_for_wb_token

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"not_access_token": "oops"}

        with patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=mock_resp):
            with pytest.raises(OAuthExchangeError, match="Malformed"):
                exchange_jwt_for_wb_token("fake.jwt.token", "https://api.wandb.ai")


# ---------------------------------------------------------------------------
# validate_bearer_token with MCP_AUTH_MODE=oauth (integration with exchange)
# ---------------------------------------------------------------------------


class TestValidateBearerTokenOAuthMode:
    @pytest.fixture(autouse=True)
    def _setup_keys(self):
        self.private_key, self.public_key = _generate_rsa_keypair()
        self.jwks = _make_jwks_dict(self.public_key)
        self.issuer = "https://auth.example.com"
        import wandb_mcp_server.oauth as oauth_mod

        oauth_mod._jwks_clients.clear()
        reset_oauth_config()

        from wandb_mcp_server.oauth_exchange import clear_exchange_cache

        clear_exchange_cache()

    def _mock_all(self):
        """Patch auth mode env, JWKS, and token exchange."""
        discovery_resp = MagicMock()
        discovery_resp.json.return_value = {"jwks_uri": f"{self.issuer}/.well-known/jwks.json"}
        discovery_resp.raise_for_status = MagicMock()

        jwks_resp = MagicMock()
        jwks_resp.json.return_value = self.jwks
        jwks_resp.raise_for_status = MagicMock()

        def httpx_get_side_effect(url, **kwargs):
            if "openid-configuration" in url:
                return discovery_resp
            return jwks_resp

        exchange_resp = MagicMock()
        exchange_resp.status_code = 200
        exchange_resp.json.return_value = {"access_token": "wb_at_exchanged.sig", "expires_in": 3600}

        env_patch = patch.dict(
            "os.environ",
            {
                "MCP_OAUTH_ISSUER": self.issuer,
                "MCP_OAUTH_AUDIENCE": "https://mcp.example.com",
            },
        )
        httpx_get_patch = patch("wandb_mcp_server.oauth.httpx.get", side_effect=httpx_get_side_effect)
        httpx_post_patch = patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=exchange_resp)
        return env_patch, httpx_get_patch, httpx_post_patch

    @pytest.mark.asyncio
    async def test_jwt_exchanged_in_oauth_mode(self):
        """JWT is validated then exchanged -- returned value is wb_at_*, not the JWT."""
        from fastapi.security import HTTPAuthorizationCredentials

        from wandb_mcp_server.auth import MCPAuthConfig, validate_bearer_token

        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
            },
            self.private_key,
        )
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        env_p, get_p, post_p = self._mock_all()
        with env_p, get_p, post_p, patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "oauth"):
            reset_oauth_config()
            result = await validate_bearer_token(creds, MCPAuthConfig())
            assert result == "wb_at_exchanged.sig"
            assert result != token

    @pytest.mark.asyncio
    async def test_api_key_fallback_in_oauth_mode(self):
        """Non-JWT tokens fall back to API key validation in OAuth mode."""
        from fastapi.security import HTTPAuthorizationCredentials

        from wandb_mcp_server.auth import MCPAuthConfig, validate_bearer_token

        api_key = "a" * 40
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=api_key)

        with patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "oauth"):
            result = await validate_bearer_token(creds, MCPAuthConfig())
            assert result == api_key

    @pytest.mark.asyncio
    async def test_jwt_rejected_in_api_key_mode(self):
        """In api-key mode, JWT-shaped tokens should fail format validation."""
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from wandb_mcp_server.auth import MCPAuthConfig, validate_bearer_token

        token = _encode_jwt(
            {"sub": "user-123", "iss": self.issuer, "exp": int(time.time()) + 3600},
            self.private_key,
        )
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "api-key"):
            with pytest.raises(HTTPException) as exc_info:
                await validate_bearer_token(creds, MCPAuthConfig())
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_exchange_failure_returns_401_not_jwt(self):
        """If token exchange fails, return 401 -- do not pass the JWT through."""
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from wandb_mcp_server.auth import MCPAuthConfig, validate_bearer_token

        token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
            },
            self.private_key,
        )
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        env_p, get_p, _ = self._mock_all()

        exchange_fail = MagicMock()
        exchange_fail.status_code = 500
        exchange_fail.text = "internal error"

        with (
            env_p,
            get_p,
            patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=exchange_fail),
            patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "oauth"),
        ):
            reset_oauth_config()
            with pytest.raises(HTTPException) as exc_info:
                await validate_bearer_token(creds, MCPAuthConfig())
            assert exc_info.value.status_code == 401
            assert "exchange" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_401_response_does_not_leak_exception_details(self):
        """Error detail must be generic, not expose internal exception strings."""
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from wandb_mcp_server.auth import MCPAuthConfig, validate_bearer_token

        expired_token = _encode_jwt(
            {
                "sub": "user-123",
                "iss": self.issuer,
                "aud": "https://mcp.example.com",
                "exp": int(time.time()) - 100,
            },
            self.private_key,
        )
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=expired_token)

        env_p, get_p, post_p = self._mock_all()
        with env_p, get_p, post_p, patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "oauth"):
            reset_oauth_config()
            with pytest.raises(HTTPException) as exc_info:
                await validate_bearer_token(creds, MCPAuthConfig())
            assert exc_info.value.status_code == 401
            assert (
                "expired" not in exc_info.value.detail.lower() or "invalid or expired" in exc_info.value.detail.lower()
            )
            assert "ExpiredSignature" not in exc_info.value.detail


# ---------------------------------------------------------------------------
# Bearer transport layer (api_client.py, weave_api/client.py)
# ---------------------------------------------------------------------------


class TestBearerTransport:
    """Tests for credential-aware auth transport selection."""

    def test_is_wb_access_token_positive(self):
        from wandb_mcp_server.api_client import is_wb_access_token

        assert is_wb_access_token("wb_at_abc123.sig456") is True

    def test_is_wb_access_token_negative_api_key(self):
        from wandb_mcp_server.api_client import is_wb_access_token

        assert is_wb_access_token("a" * 40) is False

    def test_is_wb_access_token_negative_jwt(self):
        from wandb_mcp_server.api_client import is_wb_access_token

        assert is_wb_access_token("eyJhbGciOi.eyJzdWIiOi.signature") is False

    def test_bearer_token_auth_sets_header(self):
        from wandb_mcp_server.api_client import _BearerTokenAuth

        auth = _BearerTokenAuth("wb_at_test.sig")
        r = MagicMock()
        r.headers = {}
        result = auth(r)
        assert result.headers["Authorization"] == "Bearer wb_at_test.sig"

    def test_get_api_uses_bearer_for_wb_at(self):
        """wb_at_* credential -> wandb.Api with Bearer transport, not Basic."""
        from wandb_mcp_server.api_client import WandBApiManager, _BearerTokenAuth

        with patch("wandb_mcp_server.api_client.wandb.Api") as mock_api_cls:
            mock_api = MagicMock()
            mock_api._base_client.transport.session.auth = None
            mock_api_cls.return_value = mock_api

            WandBApiManager.get_api(api_key="wb_at_real_token.hmac_sig")

            mock_api_cls.assert_called_once()
            call_kwargs = mock_api_cls.call_args
            assert call_kwargs.kwargs.get("api_key") is not None or call_kwargs.args

            assert isinstance(mock_api._base_client.transport.session.auth, _BearerTokenAuth)
            assert mock_api.api_key == "wb_at_real_token.hmac_sig"

    def test_get_api_uses_basic_for_regular_key(self):
        """Regular API key -> wandb.Api(api_key=...) with standard Basic auth."""
        from wandb_mcp_server.api_client import WandBApiManager

        regular_key = "a" * 40
        with patch("wandb_mcp_server.api_client.wandb.Api") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api

            WandBApiManager.get_api(api_key=regular_key)

            mock_api_cls.assert_called_once_with(api_key=regular_key, overrides={"base_url": "https://api.wandb.ai"})

    def test_weave_client_bearer_for_wb_at(self):
        """WeaveApiClient sends Bearer header for wb_at_* tokens."""
        from wandb_mcp_server.weave_api.client import WeaveApiClient

        client = WeaveApiClient(api_key="wb_at_test.sig", server_url="https://trace.wandb.ai")
        headers = client._get_auth_headers()
        assert headers["Authorization"] == "Bearer wb_at_test.sig"
        assert "Basic" not in headers["Authorization"]

    def test_weave_client_basic_for_api_key(self):
        """WeaveApiClient sends Basic header for regular API keys."""
        from wandb_mcp_server.weave_api.client import WeaveApiClient

        client = WeaveApiClient(api_key="a" * 40, server_url="https://trace.wandb.ai")
        headers = client._get_auth_headers()
        assert headers["Authorization"].startswith("Basic ")
        assert "Bearer" not in headers["Authorization"]
