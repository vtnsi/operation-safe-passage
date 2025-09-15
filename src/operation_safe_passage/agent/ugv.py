from __future__ import annotations
from typing import Optional, Union
from operation_safe_passage.agent.agent import Agent


class UGV(Agent):

    def __init__(
        self,
        traversal_time: int = 1,
        mineclear_time: int = 1,
        name: Optional[str] = "",
        current_node: Optional[str] = None
    ):
        """
        Creates the UGV object

        Args:
            start_node (Optional[str], optional): The start node of the UGV. Defaults to None.
            target_node (Optional[str], optional): The target node of the UGV. Defaults to None.
        """
        self.mine_nodes: set[str] = set()
        self.failures: list[dict] = []
        self.mineclear_time = mineclear_time
        super().__init__(type="UGV", traversal_time=traversal_time, name=name, current_node=current_node)

    def reset(self):
        """
        Resets the UGV
        """
        super().reset()
        self.mine_nodes.clear()
        self.failures.clear()

    def step(self, action: Union[int, str, None], node: Optional[dict]) -> bool:
        """
        Updates agent position and *inspects the passed node*:
        if a mine is present in ground truth, record it to `mine_nodes`
        and log an entry in `failures`. Also marks node['ugv_mine_detected']=True.

        Args:
            action (Union[int, str, None]): The action to take
            node (dict): Full node dictionary (must include 'id' and 'metadata').\
        Returns
            (bool): True if a mine was detected on this move.
        """
        super().step(action, node)
        if not node:
            return False
        has_mine = bool(node["inaccessible"]["mine_presence"])

        if has_mine:
            node_id = node.get("id", self.current_node)
            if node_id not in self.mine_nodes:
                self.mine_nodes.add(node_id)
            self.failures.append({"type": "mine", "node": node_id})
            node["ugv_mine_detected"] = True
        return has_mine
