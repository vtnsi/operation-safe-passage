from __future__ import annotations
from typing import Optional, Union


class Agent:
    """
    Minimal data-class style agent. No environment I/O here.
    """

    def __init__(
            self,
            type: str = "",
            traversal_time: int = 1,
            name: Optional[str] = "",
            current_node: Optional[str] = None
        ):

        self.type = type
        self.name = name
        self.traversal_time = traversal_time
        self.origin_node = current_node
        self.current_node = current_node

        self.previous_node: Optional[str] = None
        self.previous_action: Optional[str] = None

        self.scanned_nodes: set[str] = set()
        self.traversed_nodes: list[str] = []
        self.num_moves: int = 0

        self.reset()

    def reset(self) -> None:
        self.previous_node = None
        self.previous_action = None
        self.scanned_nodes.clear()
        self.current_node = self.origin_node
        self.traversed_nodes = [self.current_node] if self.current_node else []
        self.num_moves = 0

    def step(self, action: Union[int, str, None], node: Optional[Union[str, dict]]) -> None:
        """
        Controller calls this after validating the move.
        `node` may be a node-id (str) or a full node dict with an 'id' field.
        """
        self.previous_action = action
        if node is not None:
            self.move_agent(node)

    def move_agent(self, node_or_id: Union[str, dict]) -> None:
        """Accept a node id string or the full node dict; update tracking."""
        if isinstance(node_or_id, dict):
            node_id = node_or_id.get("id")
        else:
            node_id = node_or_id

        self.previous_node = self.current_node
        self.current_node = node_id
        self.traversed_nodes.append(self.current_node)
        self.num_moves += 1
