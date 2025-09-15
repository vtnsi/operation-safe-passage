"""Array-only Gymnasium environment wrapping MissionController."""

from __future__ import annotations
import numpy as np
import gymnasium as gym   # type: ignore
from gymnasium import spaces  # type: ignore
from typing import Callable, Optional
from operation_safe_passage.controller.mission_controller import MissionController


class OSPGym(gym.Env):
    """
    Gymnasium wrapper for MissionController (array-only).

    Action (MultiDiscrete):
        [ U1_move(0..6), U1_scan(0..S1), U2_move, U2_scan, ..., G1_move(0..6), G2_move, ... ]
        where:
          - move: 0..5 -> direction name (controller.dir_names), 6 -> noop
          - scan: 0..S_i-1 -> scanner i, S_i -> no-scan

    Observation (flat Box[float32]):
        For each agent (controller order), concatenated:
          [ time,
            distance_to_goal,
            distance_to_goal_weighted (or -1),
            current.weight,
            current.uav_estimate (or -1),
            current.temperature, current.wind_speed, current.visibility, current.precipitation,
            neighbor_weight[E,NE,NW,W,SW,SE] (missing -> -1),
            distances_to_other_agents (padded to (#agents-1) with -1)
          ]
    """

    metadata = {"render_modes": ["ansi"]}
    NOOP_MOVE_IDX = 6  # array action convention: 0..5 = dir, 6 = noop

    def __init__(
        self,
        param_path: str = "config/params.json",
        network_path: str = "config/network.json",
        output_dir: str = "output",
        max_steps: int = 2000,
        reward_fn: Optional[Callable[[dict, bool, bool], float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the OSPGym environment and construct action/observation spaces.

        Args:
            param_path: Path to controller parameters JSON.
            network_path: Path to mission/network JSON.
            output_dir: Directory for any controller outputs.
            max_steps: Truncation cap for steps taken in this env episode.
            reward_fn: Optional custom reward: fn(state, terminated, truncated) -> float.
            seed: RNG seed forwarded to Gymnasium seeding utilities.
        """
        super().__init__()
        self.max_steps = int(max_steps)
        self._steps = 0
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Controller
        self.controller = MissionController(
            param_path=param_path,
            network_path=network_path,
            output_dir=output_dir,
        )
        self.controller.reset()

        # Build stable agent roster
        self.agents = list(self.controller.agents)

        self.agent_names: list[str] = [agent.name for agent in self.agents]
        self.uav_indices: list[int] = [i for i, agent in enumerate(self.agents) if agent.type == "UAV"]
        self.ugv_indices: list[int] = [i for i, agent in enumerate(self.agents) if agent.type == "UGV"]

        # Per-UAV scanner names (deterministic order)
        self.uav_scanner_names: list[list[str]] = [self.agents[i].list_scanners() for i in self.uav_indices]

        # ---- Spaces ----
        # Action space: MultiDiscrete in fixed order
        highs: list[int] = []
        for names in self.uav_scanner_names:  # UAVs: (move, scan)
            highs.extend([7, len(names) + 1])
        for _ in self.ugv_indices:  # UGVs: (move)
            highs.append(7)
        self.action_space = spaces.MultiDiscrete(highs, dtype=np.int64)

        # Observation space from a sample state
        init_state = self.controller.get_state()
        sample_flat = self._build_obs_array(init_state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_flat.shape, dtype=np.float32
        )

        # Reward
        self._reward_fn = reward_fn or self._default_reward

    # ---------- Gymnasium API ----------

    def reset(self, *, seed: Optional[int] = None, _options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        Reset the underlying MissionController and return the initial observation.

        Args:
            seed: Optional RNG seed to re-seed the environment.
            options: Unused; included for Gymnasium API compatibility.

        Returns:
            obs: Flat float32 observation vector for all agents (see class docstring).
            info: dict containing:
                - "state": the full MissionController state (rich dict)
                - "time": current mission time (float)
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.controller.reset()
        self._steps = 0

        state = self.controller.get_state()
        obs = self._build_obs_array(state)
        info = {"state": state, "time": state["time"]}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the mission by one step using an array-form action.

        The action vector layout is:
            [ U1_move, U1_scan, U2_move, U2_scan, ..., G1_move, G2_move, ... ]
        where move indices are 0..5 for directions, and 6 means "noop".
        For scans, 0..S_i-1 select a scanner for UAV i, and S_i means "no-scan".

        Args:
            action: MultiDiscrete action vector as described above.

        Returns:
            obs: Next flat observation.
            reward: Float reward for this transition.
            terminated: True if any UGV is at the goal.
            truncated: True if step cap reached before termination.
            info: dict with the full "state" and current "time".
        """
        # Translate flat action to controller format
        ctrl_actions = self._actions_to_controller_format(action)

        # Advance mission
        state = self.controller.step(ctrl_actions)
        self._steps += 1

        # Termination / truncation
        terminated = self._ugv_reached_goal(state)
        truncated = (self._steps >= self.max_steps) and not terminated

        # Reward
        reward = float(self._reward_fn(state, terminated, truncated))

        # Observation
        obs = self._build_obs_array(state)
        info = {"state": state, "time": state["time"]}
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """
        Return a simple ANSI string with current time and each agent's distance/weight.

        Returns:
            A newline-joined string summary, or None if nothing to render.
        """
        s = self.controller.get_state()
        lines = [f"time={s['time']}"]
        for a in s["agents"]:
            lines.append(f"{a['type']}: d_goal={a['distance_to_goal']}, w={a['current']['weight']:.2f}")
        return "\n".join(lines)

    # ---------- Helpers ----------

    def _ugv_reached_goal(self, state: dict) -> bool:
        """
        Check termination condition: any UGV has zero hex distance to goal.

        Args:
            state: The full MissionController state dict.

        Returns:
            True if at least one UGV has reached the goal; otherwise False.
        """
        for a in state["agents"]:
            if a["type"].startswith("UGV") and float(a["distance_to_goal"]) == 0.0:
                return True
        return False

    def _default_reward(self, _state: dict, terminated: bool, truncated: bool) -> float:
        """
        Default reward function with simple shaping.

        Args:
            state: The full MissionController state dict (unused in default).
            terminated: Whether the terminal goal condition was met.
            truncated: Whether the episode hit the step cap without success.

        Returns:
            Reward float: -1 per step, +1000 on success, -10 on truncation.
        """
        r = -1.0
        if terminated:
            r += 1000.0
        if truncated:
            r -= 10.0
        return r

    def _actions_to_controller_format(self, action: np.ndarray) -> dict[object, dict[str, Optional[str]]]:
        """
        Convert flat MultiDiscrete action to the controller's expected mapping.

        Args:
            action (np.ndarray): The action object to view
                Layout of `action`:
                    [ U1_move, U1_scan, U2_move, U2_scan, ..., G1_move, G2_move, ... ]

        Returns:
            A dict keyed by agent objects:
                { agent_obj: {"move": dir_name|None, "scan": scanner_name|None} }

        Raises:
            ValueError: If action length is not the expected size or indices are out of range.
        """
        result: dict[object, dict[str, Optional[str]]] = {}

        arr = np.asarray(action, dtype=np.int64).ravel()
        expected_len = 2 * len(self.uav_indices) + len(self.ugv_indices)
        if arr.size != expected_len:
            raise ValueError(f"Flat action length {arr.size} != expected {expected_len}")

        # map index -> direction name or None
        def move_name_from_idx(idx: int) -> Optional[str]:
            """
            Translate a move index to a direction string or None for noop.
            """
            if idx == self.NOOP_MOVE_IDX:
                return None
            if 0 <= idx < len(self.controller.dir_names):
                return self.controller.dir_names[idx]
            raise ValueError(f"Move index out of range: {idx}")

        k = 0
        # UAVs: (move, scan)
        for uav_pos, agent_idx in enumerate(self.uav_indices):
            agent = self.agents[agent_idx]
            move_idx = int(arr[k])
            k += 1
            scan_idx = int(arr[k])
            k += 1

            move_name = move_name_from_idx(move_idx)
            scanner_names = self.uav_scanner_names[uav_pos]
            scan_name = None if scan_idx == len(scanner_names) else scanner_names[scan_idx]

            result[agent] = {"move": move_name, "scan": scan_name}

        # UGVs: move only
        for agent_idx in self.ugv_indices:
            agent = self.agents[agent_idx]
            move_idx = int(arr[k])
            k += 1
            move_name = move_name_from_idx(move_idx)
            result[agent] = {"move": move_name, "scan": None}

        return result

    def _build_obs_array(self, state: dict) -> np.ndarray:
        """
        Build a flat float32 observation by concatenating per-agent blocks.

        For each agent, we append:
          [ time,
            distance_to_goal,
            distance_to_goal_weighted (or -1 if None),
            current.weight,
            current.uav_estimate (or -1 if missing),
            current.temperature,
            current.wind_speed,
            current.visibility,
            current.precipitation,
            neighbor_weight[E,NE,NW,W,SW,SE] in controller.dir_names order (missing -> -1),
            distances_to_other_agents (padded to len(agents)-1 with -1)
          ]

        Args:
            state: The full MissionController state dict.

        Returns:
            A 1D numpy array (dtype float32) representing the full observation.
        """
        time_val = float(state["time"])
        blocks: list[np.ndarray] = []

        for a_state in state["agents"]:
            cur = a_state["current"]
            nbrs = a_state["neighbors"]

            d_goal = float(a_state["distance_to_goal"])
            d_goal_w = a_state["distance_to_goal_weighted"]
            d_goal_w = float(d_goal_w) if d_goal_w is not None else -1.0

            # current scalars
            weight = float(cur["weight"])
            uav_est = float(cur["uav_estimate"]) if ("uav_estimate" in cur) else -1.0
            temperature = float(cur["temperature"])
            wind_speed = float(cur["wind_speed"])
            visibility = float(cur["visibility"])
            precipitation = float(cur["precipitation"])

            # neighbor weights in fixed controller.dir_names order
            nweights: list[float] = []
            for dname in self.controller.dir_names:
                if dname in nbrs:
                    nweights.append(float(nbrs[dname]["weight"]))
                else:
                    nweights.append(-1.0)

            # distances to other agents
            dists: list[float] = []
            for other in state["agents"]:
                if other is a_state:
                    continue
                d = a_state["distance_to_agents"][other["type"]]
                dists.append(float(d))
            block = np.array(
                [
                    time_val,       # 0
                    d_goal,         # 1
                    d_goal_w,       # 2
                    weight,         # 3
                    uav_est,        # 4
                    temperature,    # 5
                    wind_speed,     # 6
                    visibility,     # 7
                    precipitation,  # 8
                    *nweights,      # neighbor weights
                    *dists,         # distances to other agents (padded)
                ],
                dtype=np.float32,
            )
            blocks.append(block)
        return np.concatenate(blocks, dtype=np.float32)
