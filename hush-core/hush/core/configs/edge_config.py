"""Edge configuration types for hush-core."""

from typing import Literal
from pydantic import BaseModel


EdgeType = Literal["normal", "lookback", "condition"]


class EdgeConfig(BaseModel):
    """Configuration for edges between nodes in a workflow graph.

    Attributes:
        from_node: Source node name
        to_node: Target node name
        type: Edge type (normal, lookback, condition)
        soft: If True, this edge doesn't count toward ready_count.
              Use for branch outputs where only one branch executes.
              Created with > operator instead of >>
    """

    from_node: str
    to_node: str
    type: EdgeType = "normal"
    soft: bool = False
