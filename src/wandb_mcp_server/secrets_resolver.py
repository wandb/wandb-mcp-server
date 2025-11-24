import os
from typing import Optional

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


class SecretsResolver:
    """
    Resolve secrets from a configured provider. Currently supports only GCP.
    """

    def __init__(self, provider: str, secrets_project: Optional[str] = None):
        if not provider:
            raise ValueError("SecretsResolver requires a provider")

        self._provider = provider.lower()
        self._project = secrets_project

        if self._provider != "gcp":
            raise ValueError(f"Unsupported secrets provider: {provider}")

        if self._provider == "gcp" and not self._project:
            raise ValueError("GCP secrets provider requires a secrets_project")

        logger.info("Initialized SecretsResolver (provider=%s, project=%s)", self._provider, self._project)

    def fetch_secret(self, secret_id: str) -> bytes:
        if not secret_id:
            raise ValueError("secret_id is required")

        if self._provider == "gcp":
            try:
                from google.cloud import secretmanager  # type: ignore  # pylint: disable=import-error
            except Exception as e:
                raise RuntimeError("google-cloud-secret-manager is not installed") from e

            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self._project}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(name=name)
            payload = response.payload.data
            return payload

        # Defensive fallback (should not be reached due to constructor checks)
        raise RuntimeError(f"Unsupported provider at runtime: {self._provider}")


def get_secrets_resolver_from_env() -> Optional[SecretsResolver]:
    provider = os.environ.get("MCP_SERVER_SECRETS_PROVIDER")
    if not provider:
        return None
    project = os.environ.get("MCP_SERVER_SECRETS_PROJECT")
    return SecretsResolver(provider=provider, secrets_project=project)


