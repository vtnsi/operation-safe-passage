from __future__ import annotations
import numpy as np
from typing import Optional, Iterable, Union
from operation_safe_passage.agent.agent import Agent


class UAV(Agent):

    def __init__(
        self,
        scanners_map: Optional[dict[str, dict]] = None,
        allowed_scanners: Optional[Iterable[str]] = None,
        accuracy_config_path: str = "config/params.json",
        rng: Optional[np.random.Generator] = None,
        exclude_scanners: Optional[Iterable[str]] = None,
        terrain_coeff: Optional[dict[str, float]] = None,
        traversal_time: int = 1,
        name: Optional[str] = "",
        current_node: str = None
    ):
        """
        Construct a UAV agent and load scanner configurations.

        Args:
            start_node (str | None): Initial node id; if None, controller will seed from mission.
            target_node (str | None): Goal node id; if None, controller will seed from mission.
            accuracy_config_path (str): Path to JSON containing scanner parameters.
            rng (np.random.Generator | None): RNG for noise; defaults to a new generator.
            exclude_scanners (Iterable[str] | None): Scanner names to exclude from use.
        """
        self.rng = rng or np.random.default_rng()
        self.exclude_scanners = set(exclude_scanners or [])
        self.acc_defaults = {
            "noise_std": 0.05,
            "visibility_metric": 7.0,
            "visibility_scale": 0.55,
            "kappa": 2.0,
            "noise_scale": 3.0,
            "threshold": 0.5,
            "time": 1.0
        }
        self.scanners: dict[str, dict] = {}
        self.scan_history: dict[str, dict[str, float]] = {}

        super().__init__(type="UAV", traversal_time=traversal_time, name=name, current_node=current_node)
        self._init_scanners_from_map(scanners_map, allowed_scanners)

    # ---------- config ----------
    def _init_scanners_from_map(
        self,
        scanners_map: Optional[dict[str, dict]],
        allowed_scanners: Optional[Iterable[str]],
    ) -> None:
        """
        Create self.scanners from a provided scanners_map and restrict to allowed_scanners.
        """
        if not scanners_map:
            scanners_map = {"default": dict(self.acc_defaults)}

        # normalize + apply defaults
        norm = {}
        for name, params in scanners_map.items():
            base = dict(self.acc_defaults)
            base.update(params or {})
            norm[name] = base

        if allowed_scanners:
            allowed = set(allowed_scanners)
            norm = {k: v for k, v in norm.items() if k in allowed}

        self.scanners = norm

    def list_scanners(self) -> list[str]:
        """
        List available scanner names in deterministic order.

        Args:
            None

        Returns:
            list[str]: Scanner names.
        """
        return list(self.scanners.keys())

    def _resolve_scanner_name(self, method: Optional[Union[str, int]]) -> str:
        """
        Resolve a scanner identifier to its correct name.

        Args:
            method (str | int | None): Scanner name, index, or None for the first scanner.

        Returns:
            str: Correct scanner name.

        Raises:
            RuntimeError: If no scanners are available.
            IndexError: If an integer index is out of range.
            KeyError: If a string name is unknown.
            TypeError: If method is not one of (None, str, int).
        """
        names = self.list_scanners()
        if not names:
            raise RuntimeError("No scanners available.")
        if method is None:
            return names[0]
        if isinstance(method, int):
            if 0 <= method < len(names):
                return names[method]
            raise IndexError(f"Scanner index out of range: {method}")
        if isinstance(method, str):
            if method in self.scanners:
                return method
            raise KeyError(f"Unknown scanner: {method}")
        raise TypeError("scan method must be None, str, or int.")

    # ---------- scan API (mutates node) ----------

    def scan(self, node: dict, method: Optional[Union[str, int]] = None) -> float:
        """
        Mutates the passed node dict in-place:
        * writes estimate to node['uav_scans'][method]
        * writes accuracy to node['metadata']['uav_accuracy']
        * writes estimate to node['metadata']['uav_estimate']
        * sets node['metadata']['scanned'] = True
        * updates node['weight'] using estimate (risk-aware)

        Args:
            node (dict): Full node dictionary (must include 'id' and 'metadata').
            method (str | int | None): Scanner name or index; None selects the first scanner.

        Returns:
            float: Estimated mine probability in [0, 1]. Returns 0.0 if scan is skipped.
        """
        if not node:
            return 0.0
        if self.current_node != node["id"]:
            return 0.0
        if not self.scanners:
            return 0.0

        scanner_name = self._resolve_scanner_name(method)
        params = self.scanners[scanner_name]

        estimate, accuracy = self._scan_one(node, params)

        meta = node["metadata"]
        # store estimate & accuracy on node
        node.setdefault("uav_scans", {})[scanner_name] = estimate
        meta["uav_estimate"] = estimate
        node['inaccessible']["uav_accuracy"] = accuracy
        meta["scanned"] = True

        # update node weight using estimate
        meta["weight"] = 100 * estimate

        # track history at agent-level too (optional; controller doesn't need it)
        self.scan_history.setdefault(node["id"], {})[scanner_name] = estimate
        self.scanned_nodes.add(node["id"])

        self.previous_action = method
        return estimate, node

    def scan_idx_for_uav(self, scanner_name: str) -> int:
        """
        For a given UAV, return the scan index for its scanner list.
        If scanner_name is None -> choose first scanner if available (i.e., scan this turn).
        """
        names = list(self.scanners.keys())
        if not names:
            return 0  # no scanners at all (degenerate); env will interpret bounds
        if scanner_name is None:
            scanner_name = names[0]
        try:
            return names.index(scanner_name)
        except ValueError:
            # unknown scanner name: fallback to first
            return 0

    # ---------- internals ----------

    def _scan_one(self, node: dict, params: dict) -> float:
        """
        Compute accuracy and estimate from node metadata; update metadata accordingly.

        Args:
            meta (dict): Node's 'metadata' dictionary (terrain, weather, etc.).
            params (dict): Scanner parameter dictionary.

        Returns:
            float: Estimated mine probability in [0, 1].
        """
        meta = node["metadata"]

        if not meta:
            return 0.0

        # Compute accuracy (hidden) and estimate (visible)
        accuracy = self._compute_accuracy(
            temperature=meta["temperature"],
            wind_speed=meta["wind_speed"],
            visibility=meta["visibility"],
            precipitation=meta["precipitation"],
            terrain_type=meta["terrain"],
            params=params,
        )

        estimate = self._compute_estimate(
            accuracy=accuracy,
            ground_truth=bool(node["inaccessible"]["mine_presence"]),
            params=params,
        )
        return float(estimate), float(accuracy)

    def _compute_accuracy(
        self,
        *,
        temperature: float,
        wind_speed: float,
        visibility: float,
        precipitation: float,
        terrain_type: str,
        params: dict,
    ) -> float:
        """
        Sensor accuracy model (logistic with environmental effects + noise).

        Args:
            temperature (float): Local temperature (°C).
            wind_speed (float): Local wind speed.
            visibility (float): Visibility percentage [0..100].
            precipitation (float): Precip amount proxy [0..50].
            terrain_type (str): Terrain label ('Grassy', 'Rocky', etc.).
            params (dict): Scanner parameters (noise_std, visibility_metric, visibility_scale, terrain_coeff).

        Returns:
            float: Accuracy in [0, 1].
        """
        noise_std = float(params.get("noise_std", self.acc_defaults["noise_std"]))
        k_vis = float(params.get("visibility_metric", self.acc_defaults["visibility_metric"]))
        v0 = float(params.get("visibility_scale", self.acc_defaults["visibility_scale"]))

        vis_scaled = visibility / 100.0
        vis_eff = 1.0 / (1.0 + np.exp(-k_vis * (vis_scaled - v0)))

        temp_eff = (1.0 - (temperature - (-10.0)) / (45.0 - (-10.0))) * vis_eff
        wind_eff = (1.0 - wind_speed / 100.0) * vis_eff
        precip_eff = np.exp(-2.0 * (precipitation / 50.0)) * vis_eff

        terrain_coeff = params.get("terrain_coeff",
                                   {"Grassy": 1.0, "Rocky": 0.5, "Sandy": 0.0, "Wooded": -0.25, "Swampy": -0.75})
        terrain_eff = terrain_coeff.get(terrain_type, 0.0) * vis_eff

        intercept, w_vis, w_temp, w_wind, w_precip, w_terrain = -2.0, 1.5, 1.0, 1.0, 1.0, 1.5
        logit = (intercept
                 + w_vis * vis_eff
                 + w_temp * temp_eff
                 + w_wind * wind_eff
                 + w_precip * precip_eff
                 + w_terrain * terrain_eff)

        acc = 1.0 / (1.0 + np.exp(-logit))
        acc += self.rng.normal(0.0, noise_std)
        return float(np.clip(acc, 0.0, 1.0))

    def _compute_estimate(self, *, accuracy: float, ground_truth: bool, params: dict) -> float:
        """
        Map accuracy and ground-truth to a probabilistic estimate via noisy logistic link.

        Args:
            accuracy (float): Sensor accuracy in [0, 1].
            ground_truth (bool): Ground truth presence of a mine (hidden to agent/UI).
            params (dict): Scanner parameters (kappa, noise_scale, threshold).

        Returns:
            float: Estimated mine probability in [0, 1].
        """
        a = float(np.clip(accuracy, 0.0, 1.0))
        kappa = float(params.get("kappa", self.acc_defaults["kappa"]))
        noise_scale = float(params.get("noise_scale", self.acc_defaults["noise_scale"]))
        threshold = float(params.get("threshold", self.acc_defaults["threshold"]))

        gt_sign = 1.0 if ground_truth else -1.0
        effective = 0.0 if a <= threshold else (a - threshold) / (1.0 - threshold)
        mu = kappa * effective * gt_sign
        sigma2 = noise_scale * (1.0 - effective)
        L = mu + self.rng.normal(0.0, np.sqrt(sigma2))
        return float(1.0 / (1.0 + np.exp(-L)))
