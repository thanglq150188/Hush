"""Logging theme and highlighters."""

from rich.theme import Theme


# Clean logging theme - optimized for readability
# INFO is neutral (default), only problems stand out
LOGGING_THEME = Theme({
    # Log levels - only highlight problems
    "logging.level.debug": "#6e7681",           # Gray (fade into background)
    "logging.level.info": "white",              # White
    "logging.level.warning": "#d29922",         # Yellow (attention)
    "logging.level.error": "#f85149",           # Red (problem)
    "logging.level.critical": "bold reverse #b81c1c",  # DANGER - white on dark red background

    # Log components - muted metadata
    "log.time": "dim white",                    # Dim white timestamp (like beeflow)
    "log.message": "white",                      # White content
    "log.path": "#6e7681",                      # Gray path

    # Custom markup colors for inline highlighting (softer/lighter versions)
    "info": "#a5d6ff",                          # Light blue
    "warning": "#f0c674",                       # Light yellow
    "error": "#ffa198",                         # Light red
    "critical": "#ffa198",                      # Light red
    "success": "#a5d6a7",                       # Light green
    "muted": "#b0b8c1",                         # Light gray
    "highlight": "#ffb86c",                     # Light orange
    "special": "#d8a9ff",                       # Light magenta
})
