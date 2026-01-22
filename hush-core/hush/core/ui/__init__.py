"""Hush Trace Viewer UI.

A web-based viewer for Hush workflow traces.

Usage:
    python -m hush.core.ui.server

Environment variables:
    HUSH_TRACES_DB: Path to traces database (default: ~/.hush/traces.db)
    HUSH_VIEWER_PORT: Server port (default: 8765)
"""

from .server import run_server

__all__ = ["run_server"]
