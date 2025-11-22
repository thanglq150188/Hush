"""Edge configuration types for hush-core."""

from typing import Literal
from pydantic import BaseModel


EdgeType = Literal["normal", "lookback", "condition"]


class EdgeConfig(BaseModel):
    """Configuration for edges between nodes in a workflow graph."""

    from_node: str
    to_node: str
    type: EdgeType = "normal"
