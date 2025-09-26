"""
Unified API client management for W&B operations.

This module provides a consistent pattern for managing W&B API instances
with per-request API keys, following the same pattern as WeaveApiClient.
"""

import os
from typing import Optional, Dict, Any
from contextvars import ContextVar
import wandb
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

# Context variable for storing the current request's API key
api_key_context: ContextVar[Optional[str]] = ContextVar('wandb_api_key', default=None)


class WandBApiManager:
    """
    Manages W&B API instances with per-request API keys.
    
    This class follows the same pattern as WeaveApiClient, providing
    a consistent interface for all W&B operations that need API access.
    """
    
    @staticmethod
    def get_api_key() -> Optional[str]:
        """
        Get the API key for the current request context.
        
        Returns:
            The API key from context, environment, or None.
        """
        # First try context variable (set by middleware for HTTP requests)
        api_key = api_key_context.get()
        
        # Fallback to environment variable (for STDIO or testing)
        if not api_key:
            api_key = os.environ.get("WANDB_API_KEY")
        
        return api_key
    
    @staticmethod
    def get_api(api_key: Optional[str] = None) -> wandb.Api:
        """
        Get a W&B API instance with the specified or current API key.
        
        Args:
            api_key: Optional API key to use. If not provided, uses context or environment.
            
        Returns:
            A configured wandb.Api instance.
            
        Raises:
            ValueError: If no API key is available.
        """
        if api_key is None:
            api_key = WandBApiManager.get_api_key()
        
        if not api_key:
            raise ValueError(
                "No W&B API key available. Provide api_key parameter or "
                "ensure WANDB_API_KEY is set in environment or request context."
            )
        
        # Create API instance with the specific key
        # According to docs: https://docs.wandb.ai/ref/python/public-api/
        return wandb.Api(api_key=api_key)
    
    @staticmethod
    def set_context_api_key(api_key: str) -> Any:
        """
        Set the API key in the current context.
        
        Args:
            api_key: The API key to set.
            
        Returns:
            A token that can be used to reset the context.
        """
        return api_key_context.set(api_key)
    
    @staticmethod
    def reset_context_api_key(token: Any) -> None:
        """
        Reset the API key context.
        
        Args:
            token: The token returned from set_context_api_key.
        """
        api_key_context.reset(token)


def get_wandb_api(api_key: Optional[str] = None) -> wandb.Api:
    """
    Convenience function to get a W&B API instance.
    
    This is the primary function that should be used throughout the codebase
    to get a W&B API instance with proper API key handling.
    
    Args:
        api_key: Optional API key. If not provided, uses context or environment.
        
    Returns:
        A configured wandb.Api instance.
    """
    return WandBApiManager.get_api(api_key)
