"""Unit tests for OAuth 2.1 token validation (oauth.py).

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
    is_jwt,
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


def _encode_jwt(claims: dict, private_key, kid="test-key-1") -> str:
    """Encode a JWT signed with the test RSA key."""
    return pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


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

    def _mock_jwks(self):
        """Patch httpx.get to return OIDC discovery + JWKS."""
        discovery_resp = MagicMock()
        discovery_resp.json.return_value = {"jwks_uri": f"{self.issuer}/.well-known/jwks.json"}
        discovery_resp.raise_for_status = MagicMock()

        jwks_resp = MagicMock()
        jwks_resp.json.return_value = self.jwks
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


# ---------------------------------------------------------------------------
# validate_bearer_token with MCP_AUTH_MODE=oauth
# ---------------------------------------------------------------------------


class TestValidateBearerTokenOAuthMode:
    @pytest.fixture(autouse=True)
    def _setup_keys(self):
        self.private_key, self.public_key = _generate_rsa_keypair()
        self.jwks = _make_jwks_dict(self.public_key)
        self.issuer = "https://auth.example.com"
        import wandb_mcp_server.oauth as oauth_mod

        oauth_mod._jwks_clients.clear()

    def _mock_all(self):
        """Patch both the auth mode env and httpx for JWKS."""
        discovery_resp = MagicMock()
        discovery_resp.json.return_value = {"jwks_uri": f"{self.issuer}/.well-known/jwks.json"}
        discovery_resp.raise_for_status = MagicMock()

        jwks_resp = MagicMock()
        jwks_resp.json.return_value = self.jwks
        jwks_resp.raise_for_status = MagicMock()

        def side_effect(url, **kwargs):
            if "openid-configuration" in url:
                return discovery_resp
            return jwks_resp

        env_patch = patch.dict(
            "os.environ",
            {
                "MCP_OAUTH_ISSUER": self.issuer,
                "MCP_OAUTH_AUDIENCE": "https://mcp.example.com",
            },
        )
        httpx_patch = patch("wandb_mcp_server.oauth.httpx.get", side_effect=side_effect)
        return env_patch, httpx_patch

    @pytest.mark.asyncio
    async def test_jwt_token_validated_in_oauth_mode(self):
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

        env_p, httpx_p = self._mock_all()
        with env_p, httpx_p, patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "oauth"):
            result = await validate_bearer_token(creds, MCPAuthConfig())
            assert result == token

    @pytest.mark.asyncio
    async def test_api_key_fallback_in_oauth_mode(self):
        from fastapi.security import HTTPAuthorizationCredentials
        from wandb_mcp_server.auth import MCPAuthConfig, validate_bearer_token

        api_key = "a" * 40
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=api_key)

        with patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "oauth"):
            result = await validate_bearer_token(creds, MCPAuthConfig())
            assert result == api_key
