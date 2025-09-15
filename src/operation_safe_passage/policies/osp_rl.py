import json
from itertools import cycle
from typing import Optional
from operation_safe_passage.controller.osp_gym import OSPGym


class OSPReinforcementLearning:
    """
    Greedy 2-phase controller driving OSPGym in **array-action** mode:
      A) UAV moves (fixed pattern E -> NE -> E -> NE ...) and scans
      B) UGV moves one step along the re-planned path to goal
    Repeats until terminated or step cap.
    """

    def __init__(
        self,
        param_path: str = "config/params.json",
        network_path: str = "config/network.json",
        max_iters: int = 2000,
        uav_scanner: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Creates an OSPReinforcementLearning object

        Args:
            param_path (str, optional): The path to the params config file. Defaults to "config/params.json".
            network_path (str, optional): The path to the network config file. Defaults to "config/network.json".
            max_iters (int, optional): The maximum number of iterations for the RL algorithm. Defaults to 2000.
            uav_scanner (Optional[str], optional): The name of the UAV scanner to use. Defaults to None.
            verbose (bool, optional): True if the algorithm should run in verbose mode. Defaults to True.

        Raises:
            RuntimeError: Thrown if the params file does not include any UGVs
        """
        self.gym = OSPGym(
            param_path=param_path,
            network_path=network_path,
            output_dir="output",
            max_steps=max_iters,
        )
        self.max_iters = int(max_iters)
        self.verbose = verbose
        self._scanner_override = uav_scanner

        # convenience handles
        self.ctrl = self.gym.controller
        self.G = self.ctrl.env.network
        self.goal = self.ctrl.end_node

        # pick the first UAV and first UGV (by type)
        self.uav = next((agent for agent in self.ctrl.agents if agent.type == "UAV"), None)
        self.ugv = next((agent for agent in self.ctrl.agents if agent.type == "UGV"), None)
        if self.ugv is None:
            raise RuntimeError("No UGV present; cannot navigate to goal.")

        # indices for packing the flat action vector (in controller order)
        self.uav_index = self.ctrl.agents.index(self.uav) if self.uav is not None else None
        self.ugv_index = self.ctrl.agents.index(self.ugv)

        # fixed UAV movement cycle: E, NE, E, NE, ...
        self._uav_dir_cycle = cycle(["E", "NE"])

        # scanner choice for the UAV (if any)
        self.scanner = None
        if self.uav is not None:
            names = self.uav.list_scanners()
            self.scanner = self._scanner_override or (names[0] if names else None)

    # ---------- helpers (packing) ----------

    def _move_idx(self, dir_name: Optional[str]) -> int:
        """
        Convert direction name to index; None -> noop.

        Args:
            dir_name (str): The str version of the direction to move

        Returns:
            (int) The direction index or numerical value
        """
        if dir_name is None:
            return self.gym.NOOP_MOVE_IDX
        return self.ctrl.dir_names.index(dir_name)

    def _build_array_action(
        self,
        uav_move_dir: Optional[str],
        uav_scan_name: Optional[str],
        ugv_move_dir: Optional[str],
    ) -> list[int]:
        """
        Build the flat MultiDiscrete action vector in OSPGym's order:
          [ (uav1_move, uav1_scan), (uav2_move, uav2_scan), ..., (ugv1_move), (ugv2_move), ... ]
        Non-acting agents receive noop and "no-scan".

        Args:
            uav_move_dir (str): The direction for the UAV to move
            uav_scan_name (str): The name of the scanner the UAV should use
            ugv_move_dir (str): The direction for the UGV to move
        
        Returns:
            (list[int]): The action built from str values
        """
        actions: list[int] = []

        # UAVs: (move, scan) for each UAV in controller order
        for idx, agent in enumerate(self.ctrl.agents):
            if agent.type != "UAV":
                continue
            if self.uav_index is not None and idx == self.uav_index:
                actions.append(self._move_idx(uav_move_dir))
                names = agent.list_scanners()
                if uav_scan_name is None:
                    actions.append(len(names))  # encode "no-scan"
                else:
                    try:
                        scan_index = names.index(uav_scan_name)
                    except ValueError:
                        raise ValueError(
                            f"Scanner {uav_scan_name!r} not available on {agent.name}. Options: {names}"
                        )
                    actions.append(scan_index)
            else:
                # other UAVs do nothing
                actions.append(self.gym.NOOP_MOVE_IDX)
                names = agent.list_scanners()
                actions.append(len(names))  # "no-scan"

        # UGVs: move only
        for idx, agent in enumerate(self.ctrl.agents):
            if agent.type != "UGV":
                continue
            if idx == self.ugv_index:
                actions.append(self._move_idx(ugv_move_dir))
            else:
                actions.append(self.gym.NOOP_MOVE_IDX)

        return actions

    def run(self):
        """
        Runs the reinforcement learning algorithm on the OSPGym
        """
        _obs, info = self.gym.reset()
        if self.verbose:
            print("Initial state:")
            print(json.dumps(info["state"], indent=4))

        steps = 0
        while steps < self.max_iters:
            steps += 1
            stop = False

            for phase in range(2):  # 0: UAV moves+scans, 1: UGV moves
                cur = self.ugv.current_node
                # Path plan from *current* UGV position
                path = self.ctrl.env.plan_path(cur, self.goal)
                next_id = path[1]
                ugv_move_dir = self.ctrl.dir_from_step(cur, next_id)

                if phase == 0:
                    # Phase A: UAV moves in fixed pattern (E, NE, ...) + scans; UGV noops
                    uav_move_dir = next(self._uav_dir_cycle) if self.uav is not None else None
                    action = self._build_array_action(
                        uav_move_dir=uav_move_dir,
                        uav_scan_name=self.scanner,  # set to None to skip scanning
                        ugv_move_dir=None,           # noop this substep
                    )
                else:
                    # Phase B: UGV moves, UAV noops
                    action = self._build_array_action(
                        uav_move_dir=None,
                        uav_scan_name=None,
                        ugv_move_dir=ugv_move_dir,
                    )

                _obs, _reward, terminated, truncated, info = self.gym.step(action)

                if self.verbose:
                    if phase == 0:
                        print(f"[{steps}A] UAV -> {uav_move_dir}, scan='{self.scanner}', time={info['time']}")
                    else:
                        d_goal = float(self.ctrl.env._hex_distance(self.ugv.current_node, self.goal))
                        print(f"[{steps}B] UGV -> {ugv_move_dir} | dist_to_goal={d_goal}, time={info['time']}")

                if terminated or truncated:
                    stop = True
                    break
            if stop:
                break

        # Final
        if self.verbose:
            print("\nFinal state:")
            # Get the latest rich state from the controller
            final_state = self.ctrl.get_state()
            print(json.dumps(final_state, indent=4))
        if self.ugv.current_node == self.goal:
            print(f"SUCCESS: {self.ugv.name} reached goal {self.goal} in {steps} planning steps.")
        else:
            print(f"STOP: iteration cap reached ({steps}) before reaching goal {self.goal}.")
