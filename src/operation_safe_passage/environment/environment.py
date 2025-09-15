import os
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from matplotlib.lines import Line2D
from typing import Iterable, Optional


class Environment:
    def __init__(
        self,
        nodes: list[dict],
        default_edge_weight: int = 50,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize and build the graph from a node-centric JSON.

        Args:
            json_path (str): Path under 'config/' or absolute path to the JSON produced by MapGenerator.
            default_edge_weight (int): Edge weight to assign (edges carry adjacency only).
            deduplicate_edges (bool): Remove duplicate undirected edges (robust default).
            output_dir (str | None): Directory for outputs (figures). Defaults to "<cwd>/output".
            figure_filename (str): Name of the saved visualization file.
        """

        nodes_section = nodes
        # ---- Build node attributes (terrain/env fields flattened + neighbors + inaccessible) ----
        node_attribute_pairs = []
        self.original_meta_keys = []

        for node_obj in nodes_section:
            node_id = node_obj["id"]

            # Copy environmental metadata flat onto the node
            attributes = dict()
            attributes['metadata'] = node_obj["metadata"]
            attributes['metadata']['mine_detected'] = False
            attributes['metadata']['UGV_navigated'] = False

            for key in attributes["metadata"].keys():
                if key not in self.original_meta_keys:
                    self.original_meta_keys.append(key)

            # Keep neighbor map for directional lookups
            attributes["neighbors"] = node_obj["neighbors"]

            # Normalize mines into 'inaccessible' sub-dict
            attributes["inaccessible"] = node_obj["inaccessible"]

            node_attribute_pairs.append((node_id, attributes))

        # Build graph and add nodes with attributes
        graph = nx.Graph()
        graph.add_nodes_from(node_attribute_pairs)

        # ---- Build edges from neighbor maps ----
        node_id_set = set(pair[0] for pair in node_attribute_pairs)
        edge_iter = (
            self._edge_iter(nodes_section, node_id_set, default_edge_weight=default_edge_weight)
        )

        graph.add_edges_from(edge_iter)

        # Expose for convenience
        self.network = graph
        self._edge_count = graph.number_of_edges()

        self.output_dir = output_dir

    def reset(self):
        """
        Resets the environment back to its original state
        """
        for node in self.network.nodes.values():
            current_meta = node["metadata"]
            to_remove = []
            for key in current_meta.keys():
                if key not in self.original_meta_keys:
                    to_remove.append(key)
            for key in to_remove:
                current_meta.pop(key)

    # -----Init Helper Functions--------
    # ---------- edge iterator----------
    def _edge_iter(
        self,
        nodes_section: Iterable[dict],
        node_id_set: set,
        default_edge_weight: int,
    ):
        """
        Iterartively add nodes to the north and east from 0,0 to the goal
        """
        keep_dirs = {"0", "1", "2"}
        for node_obj in nodes_section:
            u = node_obj["id"]
            neighbors_map = node_obj.get("neighbors", {})
            if not neighbors_map:
                continue
            for d_idx, v in neighbors_map.items():
                if d_idx in keep_dirs and v and v in node_id_set:
                    yield (u, v, {"weight": default_edge_weight})

    # ---------- convenience accessors ----------

    def edge_data(self, source_node_id: str, target_node_id: str):
        """
        Return the edge attribute dict for (source_node_id, target_node_id), or None if no edge exists.
        """
        return self.network.get_edge_data(source_node_id, target_node_id)

    def node_data(self, node_id: str) -> dict:
        """
        Return the node attributes dict (includes terrain/env fields, neighbors).
        """
        return self.network.nodes[node_id]

    def has_mine(self, node_id: str) -> bool:
        """
        True if the node has inaccessible.mine_presence.
        """
        return bool(self.node_data(node_id)["inaccessible"]["mine_presence"])

    def set_node_weight(self, node_id: str, weight: float) -> None:
        """
        Set the weight on a node (and mirror it in node_metadata).
        """
        self.network.nodes[node_id]["weight"] = float(weight)

    @staticmethod
    def _parse_node_id(node_id: str) -> tuple[int, int]:
        """
        Parse a node id string '(x, y)' into axial integer coordinates.

        Args:
            node_id (str): String like '(3, 5)'.

        Returns:
            (tuple): (x, y) integers.
        """
        node = node_id.strip()
        if node.startswith("(") and node.endswith(")"):
            node = node[1:-1]
        x_str, y_str = node.split(",")
        return int(x_str.strip()), int(y_str.strip())

    @classmethod
    def _hex_distance(cls, a: str, b: str) -> int:
        """
        Compute axial hex distance between two node ids.

        Args:
            a (str): Source node id '(x, y)'.
            b (str): Target node id '(x, y)'.

        Returns:
            int: Hex distance in edges.
        """
        x1, y1 = cls._parse_node_id(a)
        x2, y2 = cls._parse_node_id(b)
        return (abs(x1 - x2) + abs((x1 + y1) - (x2 + y2)) + abs(y1 - y2)) // 2

    def edge_w(self, u: str, v: str, d: dict) -> float:
        return float(self.network.nodes[v].get("weight", 50))

    def _weighted_distance(
        self,
        start: str,
        target: str,
        default: Optional[float] = None,
    ) -> Optional[float]:
        """
        Return the number of NODES along the current weighted-shortest path
        from `start` to `target`, using `plan_path`. If no path exists or either
        node is missing, return `default`.

        Note: This returns node-count, not edge-count. If you prefer hop count,
        use (len(path) - 1) instead of len(path).
        """
        if start is None or target is None:
            return default
        path = self.plan_path(start, target)
        return float(len(path))

    def plan_path(self, start: str, goal: str) -> list[str]:
        """Weighted shortest path; fall back to unweighted if needed."""
        return nx.shortest_path(self.network, start, goal, weight=self.edge_w)

    # ---------- visualization ----------

    def visualize_network(
        self,
        start_node: str,
        end_node: str,
        show_labels: bool = False,
        show_fig: bool = False
    ):
        """
        Render the hex grid and save the PNG to the provided path.

        Args:
            start_node: Node id to mark as the start (cyan square).
            end_node: Node id to mark as the end (red triangle).
            show_labels: If True, draw node id labels.
            show_fig: If True, call plt.show() instead of saving.
        """

        # --- infer positions from node ids like "(row, col)" ---
        def _parse_axial(node_id: str) -> tuple[int, int]:
            s = node_id.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            r_str, c_str = s.split(",")
            return int(r_str.strip()), int(c_str.strip())

        def _infer_positions(hex_size: float = 1.0) -> dict[str, tuple[float, float]]:
            positions: dict[str, tuple[float, float]] = {}
            for node_id in self.network.nodes():
                row, col = _parse_axial(node_id)
                x = hex_size * math.sqrt(3) * (row + col / 2.0)
                y = hex_size * (3.0 / 2.0) * col
                positions[node_id] = (x, y)
            return positions

        try:
            positions = _infer_positions()
        except Exception:
            # Fallback layout if IDs aren't parseable
            positions = nx.spring_layout(self.network, seed=42)

        terrain_colors = {
            "Grassy": "green",
            "Rocky": "gray",
            "Sandy": "yellow",
            "Wooded": "darkgreen",
            "Swampy": "brown",
        }

        plt.figure(figsize=(9, 7))
        nx.draw_networkx_edges(
            self.network,
            positions,
            edge_color="lightgray",
            width=1.5,
        )

        # Special nodes for styling (guard against None / missing members)
        special_nodes = set()
        if start_node and start_node in self.network:
            special_nodes.add(start_node)
        if end_node and end_node in self.network:
            special_nodes.add(end_node)

        normal_nodes = [nid for nid in self.network.nodes() if nid not in special_nodes]

        # Node colors by terrain (fallback to lightgray if missing)
        node_colors = []
        node_edgecolors = []
        present_terrains = set()

        for nid in normal_nodes:
            meta = self.network.nodes[nid].get("metadata", {})
            terr = meta.get("terrain")
            if terr is not None:
                present_terrains.add(terr)
            node_colors.append(terrain_colors.get(terr, "lightgray"))

            # Robust mine outline detection:
            attrs = self.network.nodes[nid]
            mine_flag = bool(attrs.get("inaccessible", {}).get("mine_presence", False)) \
                        or bool(attrs.get("metadata", {}).get("mine", False))
            node_edgecolors.append("black" if mine_flag else "white")

        nx.draw_networkx_nodes(
            self.network,
            positions,
            nodelist=normal_nodes,
            node_color=node_colors,
            edgecolors=node_edgecolors,
            linewidths=1.5,
            node_size=150,
        )

        # Start node (square)
        if start_node and start_node in self.network:
            nx.draw_networkx_nodes(
                self.network,
                positions,
                nodelist=[start_node],
                node_color="cyan",
                edgecolors="black",
                linewidths=2.0,
                node_size=260,
                node_shape="s",
            )

        # End node (triangle)
        if end_node and end_node in self.network:
            nx.draw_networkx_nodes(
                self.network,
                positions,
                nodelist=[end_node],
                node_color="red",
                edgecolors="black",
                linewidths=2.0,
                node_size=260,
                node_shape="^",
            )

        if show_labels:
            nx.draw_networkx_labels(
                self.network,
                positions,
                labels={nid: nid for nid in self.network.nodes()},
                font_size=6,
            )

        plt.title("Hex Grid: terrain fill, black outline = mine")
        plt.axis("off")

        # ------- Legend (only show terrains that actually appear) -------
        terrain_patches = []
        for label, color in terrain_colors.items():
            if label in present_terrains:
                terrain_patches.append(mpatches.Patch(color=color, label=label))

        mine_patch = mpatches.Patch(facecolor="none", edgecolor="black", label="Mine (outline)")
        start_handle = Line2D([0], [0], marker="s", color="w", markerfacecolor="cyan",
                              markeredgecolor="black", markersize=10, label="Start")
        end_handle = Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
                            markeredgecolor="black", markersize=10, label="End")

        legend_handles = terrain_patches + [mine_patch]
        if start_node and start_node in self.network:
            legend_handles.append(start_handle)
        if end_node and end_node in self.network:
            legend_handles.append(end_handle)

        if legend_handles:
            plt.legend(handles=legend_handles, loc="lower right")

        if show_fig:
            return plt.show()

        plt.savefig(
            os.path.join(self.output_dir, "generated_map.png"),
            bbox_inches="tight",
            dpi=200
        )
        plt.close()
