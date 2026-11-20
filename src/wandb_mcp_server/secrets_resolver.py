"""Bounded secret resolution with diagnostics safe for shared logs."""

import os
import re
from typing import Optional

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)
SECRET_ATTEMPT_TIMEOUT_SECONDS = 3.0


class SecretResolutionError(RuntimeError):
    """Secret dependency failure with a fixed, non-sensitive reason."""

    _REASONS = frozenset({"configuration", "access_denied", "not_found", "disabled", "timeout", "unavailable", "empty"})

    def __init__(self, reason: str):
        self.reason = reason if reason in self._REASONS else "unavailable"
        super().__init__(f"Secret dependency unavailable ({self.reason})")


class SecretsResolver:
    """Resolve a secret version from GCP without implicit SDK retries."""

    def __init__(self, provider: str, secrets_project: Optional[str] = None):
        if not isinstance(provider, str) or provider.lower() != "gcp" or not secrets_project:
            raise SecretResolutionError("configuration")
        self._provider = provider.lower()
        self._project = secrets_project
        logger.info("Initialized SecretsResolver (provider=%s)", self._provider)

    def fetch_secret(self, secret_id: str, *, version: str = "latest") -> bytes:
        if (
            not isinstance(secret_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,255}", secret_id) is None
            or not isinstance(version, str)
            or re.fullmatch(r"latest|[1-9][0-9]{0,19}", version) is None
        ):
            raise SecretResolutionError("configuration")
        try:
            from google.api_core import exceptions  # type: ignore
            from google.cloud import secretmanager  # type: ignore
        except ImportError:
            raise SecretResolutionError("configuration") from None

        try:
            with secretmanager.SecretManagerServiceClient() as client:
                name = f"projects/{self._project}/secrets/{secret_id}/versions/{version}"
                for attempt in range(2):
                    try:
                        response = client.access_secret_version(
                            name=name, retry=None, timeout=SECRET_ATTEMPT_TIMEOUT_SECONDS
                        )
                        payload = response.payload.data
                        if not isinstance(payload, bytes) or not payload:
                            raise SecretResolutionError("empty")
                        return payload
                    except (exceptions.DeadlineExceeded, exceptions.ServiceUnavailable, exceptions.TooManyRequests):
                        if attempt == 1:
                            raise
        except SecretResolutionError:
            raise
        except (exceptions.PermissionDenied, exceptions.Unauthenticated):
            raise SecretResolutionError("access_denied") from None
        except exceptions.NotFound:
            raise SecretResolutionError("not_found") from None
        except exceptions.FailedPrecondition:
            raise SecretResolutionError("disabled") from None
        except exceptions.DeadlineExceeded:
            raise SecretResolutionError("timeout") from None
        except Exception:
            raise SecretResolutionError("unavailable") from None
        raise SecretResolutionError("unavailable")


def get_secrets_resolver_from_env() -> Optional[SecretsResolver]:
    provider = os.environ.get("MCP_SERVER_SECRETS_PROVIDER")
    if not provider:
        return None
    project = os.environ.get("MCP_SERVER_SECRETS_PROJECT")
    return SecretsResolver(provider=provider, secrets_project=project)
