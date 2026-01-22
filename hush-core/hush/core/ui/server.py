"""Simple HTTP server for Hush Trace Viewer.

Serves the trace viewer HTML and provides an API endpoint to read the SQLite database.

Usage:
    python -m hush.core.ui.server
    # or
    python hush/core/ui/server.py

Environment variables:
    HUSH_TRACES_DB: Path to traces database (default: ~/.hush/traces.db)
    HUSH_VIEWER_PORT: Server port (default: 8765)
"""

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# Default paths
DEFAULT_DB_PATH = Path.home() / ".hush" / "traces.db"
DEFAULT_PORT = 8765


class TraceViewerHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves static files and the database."""

    def __init__(self, *args, db_path: Path, **kwargs):
        self.db_path = db_path
        # Set directory to the ui folder
        self.directory = str(Path(__file__).parent)
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/db":
            self.serve_database()
        elif parsed.path == "/api/status":
            self.serve_status()
        elif parsed.path == "/api/debug":
            self.serve_debug()
        elif parsed.path == "/favicon.ico":
            # Return empty favicon to avoid 404 errors
            self.send_response(204)
            self.end_headers()
        elif parsed.path == "/" or parsed.path == "":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def serve_database(self):
        """Serve the SQLite database file."""
        if not self.db_path.exists():
            self.send_error(404, f"Database not found: {self.db_path}")
            return

        try:
            with open(self.db_path, "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", len(data))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def serve_status(self):
        """Serve status info including DB path and existence."""
        import json

        status = {
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
            "db_size": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

        data = json.dumps(status).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def serve_debug(self):
        """Serve debug info about traces in the database."""
        import json
        import sqlite3

        if not self.db_path.exists():
            self.send_error(404, "Database not found")
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            # Get sample traces to debug tree structure
            cursor = conn.execute("""
                SELECT id, request_id, node_name, parent_name, context_id, execution_order, duration_ms
                FROM traces
                ORDER BY created_at DESC
                LIMIT 50
            """)

            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()

            data = json.dumps(rows, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        """Custom log format."""
        if args and isinstance(args[0], str) and "/api/" in args[0]:
            print(f"[TraceViewer] {args[0]}")


def make_handler(db_path: Path):
    """Create handler class with db_path bound."""
    def handler(*args, **kwargs):
        return TraceViewerHandler(*args, db_path=db_path, **kwargs)
    return handler


def run_server(db_path: Path = None, port: int = None):
    """Run the trace viewer server.

    Args:
        db_path: Path to SQLite database. Defaults to HUSH_TRACES_DB env or ~/.hush/traces.db
        port: Server port. Defaults to HUSH_VIEWER_PORT env or 8765
    """
    db_path = db_path or Path(os.environ.get("HUSH_TRACES_DB", DEFAULT_DB_PATH))
    port = port or int(os.environ.get("HUSH_VIEWER_PORT", DEFAULT_PORT))

    handler = make_handler(db_path)
    server = HTTPServer(("localhost", port), handler)

    print(f"Hush Trace Viewer")
    print(f"  Database: {db_path} ({'exists' if db_path.exists() else 'not found'})")
    print(f"  URL: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    # Allow port as command line argument
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_server(port=port)
