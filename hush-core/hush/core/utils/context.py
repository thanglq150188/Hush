import contextvars
from typing import Optional

_current_graph = contextvars.ContextVar("current_graph")


def get_current():
    try:
        return _current_graph.get()
    except LookupError:
        return None
