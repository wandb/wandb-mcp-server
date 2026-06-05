"""Helpers for importing W&B SDK vendored modules."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import wandb


@contextmanager
def wandb_vendor_path() -> Iterator[None]:
    """Temporarily expose W&B SDK vendored packages on ``sys.path``."""
    reset_path = wandb.util.vendor_setup()
    try:
        yield
    finally:
        reset_path()
