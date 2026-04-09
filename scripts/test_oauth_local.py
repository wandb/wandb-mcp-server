#!/usr/bin/env python3
"""Self-contained local OAuth 2.1 end-to-end test.

Spins up a minimal OIDC provider on localhost, signs a JWT, and
exercises the full MCP OAuth flow (validation -> exchange -> Bearer
transport selection) without any network calls to real W&B services.

Usage:
    uv run python scripts/test_oauth_local.py

All steps are mocked at the HTTP boundary -- no WANDB_API_KEY, no
staging access, no Gorilla OIDC provider registration required.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Step 0: Generate RSA key pair and build JWKS
# ---------------------------------------------------------------------------

from cryptography.hazmat.primitives.asymmetric import rsa

private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()


def _build_jwks():
    from jwt.algorithms import RSAAlgorithm

    jwk = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk["kid"] = "local-test-key"
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    return {"keys": [jwk]}


JWKS = _build_jwks()
ISSUER = None  # set after server starts

# ---------------------------------------------------------------------------
# Step 1: Local OIDC provider (discovery + JWKS)
# ---------------------------------------------------------------------------


class OIDCHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/.well-known/openid-configuration":
            body = json.dumps(
                {
                    "issuer": ISSUER,
                    "jwks_uri": f"{ISSUER}/jwks.json",
                    "token_endpoint": f"{ISSUER}/oidc/token",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/jwks.json":
            body = json.dumps(JWKS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs


def start_oidc_server():
    global ISSUER
    server = HTTPServer(("127.0.0.1", 0), OIDCHandler)
    server.timeout = 0.5
    port = server.server_address[1]
    ISSUER = f"http://127.0.0.1:{port}"
    thread = Thread(target=server.serve_forever, daemon=True, kwargs={"poll_interval": 0.25})
    thread.start()
    return server


# ---------------------------------------------------------------------------
# Step 2: Sign a JWT
# ---------------------------------------------------------------------------


def sign_jwt(claims: dict) -> str:
    import jwt as pyjwt

    return pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "local-test-key"})


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  -- {detail}")


async def run_tests():
    global passed, failed

    server = start_oidc_server()
    print(f"\nLocal OIDC provider running at {ISSUER}")
    print("=" * 60)

    # Clear module-level caches
    import wandb_mcp_server.oauth as oauth_mod

    oauth_mod._jwks_clients.clear()
    oauth_mod.reset_oauth_config()

    from wandb_mcp_server.oauth_exchange import clear_exchange_cache

    clear_exchange_cache()

    # -----------------------------------------------------------------------
    # Test 1: JWT validation against local OIDC provider
    # -----------------------------------------------------------------------
    print("\n--- Test 1: JWT validation via local JWKS ---")

    from wandb_mcp_server.oauth import OAuthConfig, validate_oauth_token

    token = sign_jwt(
        {
            "sub": "test-user-123",
            "iss": ISSUER,
            "aud": "https://mcp.test.wandb.ai",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "email": "test@wandb.com",
        }
    )

    config = OAuthConfig(issuer=ISSUER, audience="https://mcp.test.wandb.ai", required_scopes=[])
    try:
        claims = validate_oauth_token(token, config)
        check("JWT validates against local JWKS", claims["sub"] == "test-user-123")
    except Exception as e:
        check("JWT validates against local JWKS", False, str(e))

    # -----------------------------------------------------------------------
    # Test 2: Token exchange returns wb_at_* (mocked)
    # -----------------------------------------------------------------------
    print("\n--- Test 2: Token exchange -> wb_at_* ---")

    from wandb_mcp_server.oauth_exchange import exchange_jwt_for_wb_token

    clear_exchange_cache()

    mock_exchange_resp = MagicMock()
    mock_exchange_resp.status_code = 200
    mock_exchange_resp.json.return_value = {
        "access_token": "wb_at_localtest_abcdef.hmac_signature_here",
        "expires_in": 3600,
    }

    with patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=mock_exchange_resp):
        wb_token = exchange_jwt_for_wb_token(token, "http://localhost:9999")
        check("Exchange returns wb_at_* token", wb_token.token.startswith("wb_at_"))
        check("Exchange token is not the JWT", wb_token.token != token)
        check("Token is not expired", not wb_token.is_expired)

    # -----------------------------------------------------------------------
    # Test 3: Full validate_bearer_token with OAuth mode
    # -----------------------------------------------------------------------
    print("\n--- Test 3: Full auth flow (JWT -> exchange -> wb_at_*) ---")

    from fastapi.security import HTTPAuthorizationCredentials

    from wandb_mcp_server.auth import MCPAuthConfig, validate_bearer_token

    clear_exchange_cache()
    oauth_mod._jwks_clients.clear()
    oauth_mod.reset_oauth_config()

    env = {
        "MCP_OAUTH_ISSUER": ISSUER,
        "MCP_OAUTH_AUDIENCE": "https://mcp.test.wandb.ai",
    }
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with (
        patch.dict("os.environ", env),
        patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "oauth"),
        patch("wandb_mcp_server.oauth_exchange.httpx.post", return_value=mock_exchange_resp),
    ):
        oauth_mod.reset_oauth_config()
        result = await validate_bearer_token(creds, MCPAuthConfig())
        check("Bearer token returns wb_at_*", result.startswith("wb_at_"))
        check("Bearer token is NOT the raw JWT", result != token)
        check("Bearer token matches exchange response", result == "wb_at_localtest_abcdef.hmac_signature_here")

    # -----------------------------------------------------------------------
    # Test 4: API key fallback in OAuth mode
    # -----------------------------------------------------------------------
    print("\n--- Test 4: API key fallback in OAuth mode ---")

    api_key = "a" * 40
    creds_apikey = HTTPAuthorizationCredentials(scheme="Bearer", credentials=api_key)

    with patch("wandb_mcp_server.auth.MCP_AUTH_MODE", "oauth"):
        result = await validate_bearer_token(creds_apikey, MCPAuthConfig())
        check("API key passes in OAuth mode", result == api_key)
        check("API key is returned unchanged", not result.startswith("wb_at_"))

    # -----------------------------------------------------------------------
    # Test 5: Bearer transport selection
    # -----------------------------------------------------------------------
    print("\n--- Test 5: Bearer vs Basic transport selection ---")

    from wandb_mcp_server.api_client import _BearerTokenAuth, is_wb_access_token

    check("wb_at_* detected as access token", is_wb_access_token("wb_at_test.sig"))
    check("API key NOT detected as access token", not is_wb_access_token("a" * 40))

    auth = _BearerTokenAuth("wb_at_test.sig")
    mock_req = MagicMock()
    mock_req.headers = {}
    auth(mock_req)
    check("BearerTokenAuth sets Bearer header", mock_req.headers.get("Authorization") == "Bearer wb_at_test.sig")

    # -----------------------------------------------------------------------
    # Test 6: Weave client auth headers
    # -----------------------------------------------------------------------
    print("\n--- Test 6: Weave client Bearer vs Basic ---")

    from wandb_mcp_server.weave_api.client import WeaveApiClient

    weave_oauth = WeaveApiClient(api_key="wb_at_test.sig", server_url="http://localhost")
    headers_oauth = weave_oauth._get_auth_headers()
    check("Weave Bearer for wb_at_*", headers_oauth["Authorization"] == "Bearer wb_at_test.sig")

    weave_apikey = WeaveApiClient(api_key="a" * 40, server_url="http://localhost")
    headers_basic = weave_apikey._get_auth_headers()
    check("Weave Basic for API key", headers_basic["Authorization"].startswith("Basic "))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed > 0:
        print("\nFAILED -- see details above")
    else:
        print("\nALL TESTS PASSED -- OAuth flow works end-to-end locally")

    server.shutdown()
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
