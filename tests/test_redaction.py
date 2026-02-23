"""Unit tests for _redact_params() and log_tool_call() -- sensitive field redaction (MCP-10)."""

from wandb_mcp_server.mcp_tools.tools_utils import _redact_params, log_tool_call


class TestRedactParams:
    def test_redacts_api_key(self):
        assert _redact_params({"api_key": "sk-12345"})["api_key"] == "***REDACTED***"

    def test_redacts_token(self):
        assert _redact_params({"bearer_token": "abc"})["bearer_token"] == "***REDACTED***"

    def test_redacts_secret(self):
        assert _redact_params({"client_secret": "s3cr3t"})["client_secret"] == "***REDACTED***"

    def test_redacts_password(self):
        result = _redact_params({"db_password": "hunter2", "host": "localhost"})
        assert result["db_password"] == "***REDACTED***"
        assert result["host"] == "localhost"

    def test_redacts_credential(self):
        assert _redact_params({"credential_file": "/creds"})["credential_file"] == "***REDACTED***"

    def test_redacts_auth(self):
        result = _redact_params({"auth_header": "Bearer xxx", "timeout": 30})
        assert result["auth_header"] == "***REDACTED***"
        assert result["timeout"] == 30

    def test_case_insensitive(self):
        result = _redact_params({"API_KEY": "v", "Token": "v2", "SECRET_VALUE": "v3"})
        assert all(v == "***REDACTED***" for v in result.values())

    def test_no_sensitive_fields_unchanged(self):
        params = {"project": "test", "entity": "user", "limit": 10}
        assert _redact_params(params) == params

    def test_empty_params(self):
        assert _redact_params({}) == {}

    def test_log_tool_call_does_not_raise(self):
        log_tool_call("test_tool", "viewer", {"api_key": "secret", "x": 1})
