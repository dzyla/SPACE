"""
SPACE core — pure computation, no Streamlit.

Every front-end (Streamlit app, ``space_api``, the CLI and the MCP server) is a
thin layer over these modules. Functions report problems through the
``"space"`` logger and long-running steps accept an optional ``progress``
callback ``(fraction: float, message: str) -> None``.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger("space")

ProgressFn = Optional[Callable[[float, str], None]]


def report(progress: ProgressFn, fraction: float, message: str) -> None:
    """Call ``progress`` if given; never let a UI callback break computation."""
    if progress is None:
        return
    try:
        progress(max(0.0, min(1.0, fraction)), message)
    except Exception:  # noqa: BLE001
        log.debug("progress callback failed", exc_info=True)


REFERENCE_ID = "reference_sequence"
