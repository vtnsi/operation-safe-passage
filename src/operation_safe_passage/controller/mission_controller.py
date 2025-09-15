from __future__ import annotations
import json
import os
from typing import Optional
from operation_safe_passage.agent.uav import UAV
from operation_safe_passage.agent.ugv import UGV
from operation_safe_passage.environment.environment import Environment
from operation_safe_passage.environment.time_keeper import TimeKeeper
from operation_safe_passage.controller.validate_mission_params import JsonValidator


class MissionController:
    """
    Orchestrates:
      - neighbor-checked movement (then passes full node dict to agent.step)
      - time accounting for traverse / scan / clear
      - UAV scan call (agent mutates node & weight)
      - UGV mine handling time penalty (based on bool returned from ugv.step)
      - get_state(): current node + neighbors (as full node dicts), and distances
    """

    def __init__(
            self,
            param_path: str = "config/params.json",
            network_path: str = "config/network.json",
            output_dir: str = "output"):
        """
        Initialize a mission controller with an environment and agents.

        Args:
            env: Environment-like object exposing:
                 - network (networkx.Graph)
                 - node_meta(str) -> dict
                 - neighbor(node_id, dir_idx) -> Optional[str]  (optional but preferred)
                 - direction_order (list[str])  (optional)
                 - time.current_time (optional)
                 - advance_time(task_key, count=1) and add_time(seconds) (optional)
            agents (Iterable): Iterable of agent instances (e.g., UAV/UGV).

        Returns:
            None
        """
        validator = JsonValidator()
        validator.validate_config(param_path)

        self.dir_names = ["E", "NE", "NW", "W", "SW", "SE"]

        # Output directory / figure name
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.time = TimeKeeper()

        # Load network and create Environment object
        with open(network_path, "r", encoding="utf-8") as f:
            network_json = json.load(f)
        self.env = Environment(nodes=network_json["nodes"], output_dir=self.output_dir)

        self.start_node = network_json["mission"]["start"]
        self.end_node = network_json["mission"]["end"]
        self.direction_order = network_json["mission"]["direction_order"]

        # Load params and create Agents objects
        with open(param_path, "r", encoding="utf-8") as f:
            param_json = json.load(f)

        agent_conf = param_json["agents"]
        scanner_map = param_json["scanner_params"]['scanners']
        terrain_coeff = param_json["terrain_coeff"]

        # ----- build agents -----
        self.agents: list[object] = []

        # UAVs: each UAV can restrict to a subset of scanners
        uavs = agent_conf["uavs"]
        for _index, uav_spec in enumerate(uavs):
            uav = UAV(
                scanners_map=scanner_map,  # pass all scanners once
                allowed_scanners=uav_spec["scanners"],
                terrain_coeff=terrain_coeff,
                traversal_time=float(param_json["processing_params"].get("UAV traversal time", 1)),
                name=uav_spec["name"],
                current_node=self.start_node
            )
            self.agents.append(uav)

        # UGV(s)
        num_ugvs = int(agent_conf["num_ugvs"])
        for i in range(num_ugvs):
            ugv = UGV(
                traversal_time=float(param_json["processing_params"].get("UGV_traversal_time", 1)),
                mineclear_time=float(param_json["processing_params"].get("UGV_clear_time", 60)),
                name=f"UGV_{i}",
                current_node=self.start_node
            )
            self.agents.append(ugv)

    # ---------------- lifecycle ----------------

    def reset(self) -> None:
        """
        Reset all agents (seeding START/TARGET from environment if missing)
        and reset the environment time to 0 if available.
        """
        for agent in self.agents:
            agent.reset()
            agent.current_node = self.start_node
        self.env.reset()
        self.time.current_time = 0.0

    # ---------------- stepping ----------------

    def step(self, actions: dict[object, dict]) -> dict:
        """
        Advance the mission by one step for a set of agent actions.

        Args:
            actions (dict): Mapping {agent: {'move': dir|name|None, 'scan': name|index|None}}
                            - 'move' is validated against neighbors; if valid, agent is moved.
                            - 'scan' (UAV only) scans the agent's current cell (after move).
                            Time penalties are applied where available (traverse/scan/clear).

        Returns:
            dict: Aggregated mission state as returned by get_state().
        """
        for agent, spec in actions.items():
            move = spec["move"]
            scan = spec.get("scan", None)  # might not always have a scan
            # MOVE (neighbor-validated)
            moved, node_dict = self._move_agent_and_get_node(agent, move)

            # Traverse time if moved
            if moved:
                self.time.add_time(agent.traversal_time)
                # UGV: mine detect + clear time
                if agent.type == "UGV":
                    mine = agent.step(move, node_dict)  # returns True if mine present
                    node_dict['metadata']['UGV_navigated'] = True
                    if mine:
                        node_dict['metadata']['mine_detected'] = True
                        self.time.add_time(agent.mineclear_time)
                # UAV: scan current cell only
                elif scan is not None and agent.type == "UAV":
                    estimate, node = agent.scan(node_dict, scan)
                    self.time.add_time(float(agent.scanners[agent._resolve_scanner_name(scan)].get("time",
                                                                                                   agent.acc_defaults["time"])))
                    if estimate < node['metadata']['weight']:
                        self.env.set_node_weight(node_id=node_dict['id'], weight=estimate * 100)

        return self.get_state()

    # ---------------- state ----------------

    def get_state(self) -> dict:
        """
        Build and return the public mission state for all agents.

        Returns:
            dict: {
                'time': float|None,

                'agents': [
                    {
                        'type': str,
                        'current': dict|None,                 # full node dict (scrubbed)
                        'neighbors': dict[str, dict],         # dir_name -> full node dict (scrubbed)
                        'distance_to_goal': float|None,       # hex distance
                        'distance_to_goal_weighted': float|None,
                        'distance_to_ugv': float|None,        # hex distance to UGV
                        'previous_node': str|None,
                        'previous_action': str|None,
                        'num_moves': int

                    }, ...

                ]

            }
        """
        state = {"time": self.time.current_time, "agents": []}

        # Precompute all agent positions and labels
        all_positions = {agent: agent.current_node for agent in self.agents}
        all_labels = {agent: agent.name for agent in self.agents}

        for agent in self.agents:
            cur_id = agent.current_node
            # Current (scrubbed) node
            current = self._node_dict(cur_id)

            # Neighbors (scrubbed)
            neighbors = {}
            for neighbor_index, neighbor_node in (self.env.node_data(cur_id)["neighbors"].items()):
                if neighbor_node is not None:
                    neighbors[self.dir_names[int(neighbor_index)]] = self._node_dict(neighbor_node)['metadata']

            # Distances to goal (hex + weighted)
            d_goal = float(self.env._hex_distance(cur_id, self.end_node))
            d_goal_w = self.env._weighted_distance(cur_id, self.end_node)

            # NEW: distances to all other agents
            distances_to_agents = {}
            for other, other_id in all_positions.items():
                if other is agent or other_id is None:
                    continue
                distances_to_agents[all_labels[other]] = float(self.env._hex_distance(cur_id, other_id))

            state["agents"].append({
                "type": agent.name,
                "current": current['metadata'],
                "neighbors": neighbors,
                "distance_to_goal": d_goal,
                "distance_to_goal_weighted": d_goal_w,
                "distance_to_agents": distances_to_agents,
                "previous_node": agent.previous_node,
                "previous_action": agent.previous_action,
                "num_moves": agent.num_moves,
            })

        return state

    def dir_from_step(self, cur_id: str, next_id: str) -> str:
        """Map (cur->next) to direction name using neighbor table."""
        nmap = self.env.node_data(cur_id)["neighbors"]
        for idx_str, nid in nmap.items():
            if nid == next_id:
                return self.dir_names[int(idx_str)]
        raise RuntimeError(f"No neighbor from {cur_id} to {next_id}.")

    def _public_node_view(self, node_dict: dict) -> dict:
        """
        Produce a scrubbed, public version of a node dictionary.

        Args:
            node_dict (dict): Full node dict as returned by _node_dict().

        Returns:
            dict: Shallow copy with metadata cleared of hidden fields:
                  - removes metadata['inaccessible'] / legacy 'inaccessible'
                  - removes any private accuracy fields if present
        """
        out = dict(node_dict)
        out.pop("inaccessible")
        return out

    def _move_agent_and_get_node(self, agent, direction) -> tuple[bool, Optional[dict]]:
        """
        Validate a neighbor move, perform the agent step, and return the new node.

        Args:
            agent: Agent instance (must expose current_node, step()).
            direction (int | str | None): Direction index in [0..5], direction name (e.g., 'NE'),
                                          or None/'noop' for no movement.

        Returns:
            (tuple):
                moved (bool): True if the move succeeded; False if blocked or no-op.
                node_dict (dict | None): Full node dict of the *current* node after action.
        """
        # no-op
        if direction in (None, "noop"):
            agent.step("noop", None)
            return False, self._node_dict(agent.current_node)

        # resolve direction index
        try:
            if isinstance(direction, int):
                d_idx = direction if 0 <= direction <= 5 else None
            else:  # str
                d_idx = self.dir_names.index(direction)
        except ValueError:
            return False, None

        # compute neighbor
        neighbor_map = (self.env.node_data(agent.current_node)["neighbors"])
        next_id = neighbor_map.get(str(d_idx), None)

        if not next_id:
            agent.previous_action = "blocked"
            return False, self._node_dict(agent.current_node)

        node_dict = self._node_dict(next_id)
        agent.step(self.dir_names[d_idx], node_dict)
        return True, node_dict

    def _node_dict(self, node_id: str) -> dict:
        """
        Build a shallow node dictionary suitable for agents to mutate.

        Args:
            node_id (str): Node identifier like '(q, r)'.

        Returns:
            dict: Shallow copy of graph node attributes with:
                  - 'id': node_id
                  - 'metadata': ensured to exist (from env.node_meta(node_id))
        """
        attrs = dict(self.env.node_data(node_id))
        attrs["id"] = node_id
        return attrs
