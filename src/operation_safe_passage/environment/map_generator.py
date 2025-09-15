import json
import os
import networkx as nx  # type: ignore
import numpy as np


class MapGenerator:
    """
    Generate a node-centric, hex terrain map with per-node
    environmental metadata, neighbor pointers, and mine placement.
    """

    def __init__(self, parameters_file, output_dir=None, json_filename="network.json"):
        """
        Initialize the map generator from a parameter JSON file and set up outputs.

        Args:
            parameters_file (str): Path to a JSON file containing configuration keys:
                - num_nodes (int)
                - terrain_types (list[str])
                - transition_matrix (list[list[float]])
                - precipitation_params (dict)
                - temp_params (dict)
                - wind_params (dict)
                - visibility_params (dict)
                - mine_likelihood (float)
            output_dir (str, optional): Directory where outputs will be written. If None,
                uses "<cwd>/output" and creates the directory if needed.
            json_filename (str, optional): Filename for the exported JSON.
        """
        with open(parameters_file, 'r') as f:
            data = json.load(f)

        # Parameters
        params = data['environment']
        self.num_nodes = params['num_nodes']
        self.terrain_types = params['terrain_types']
        self.transition_matrix = np.array(params['transition_matrix'])
        self.precipitation_params = params['precipitation_params']
        self.temp_params = params['temp_params']
        self.wind_params = params['wind_params']
        self.visibility_params = params['visibility_params']
        self.mine_likelihood = params['mine_likelihood']

        # Outputs
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.json_filename = json_filename

        # Fixed 6-direction axial offsets (pointy-topped).
        # 0:E, 1:SE, 2:SW, 3:W, 4:NW, 5:NE
        self.DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        self.DIR_NAMES = ['E', 'NE', 'NW', 'W', 'SW', 'SE']

    def assign_terrain(self, neighbor_terrains):
        """
        Choose a terrain type conditioned on already-assigned neighbor terrains.

        Args:
            neighbor_terrains (list[str]): Terrain labels of already-assigned neighbors.

        Returns:
            str: Chosen terrain type.
        """
        if not neighbor_terrains:
            return np.random.choice(self.terrain_types)

        num_terrains = len(self.terrain_types)
        prob_distribution = np.zeros(num_terrains)
        for terrain in neighbor_terrains:
            idx = self.terrain_types.index(terrain)
            prob_distribution += self.transition_matrix[idx]
        prob_distribution /= prob_distribution.sum()
        return np.random.choice(self.terrain_types, p=prob_distribution)

    def generate_precipitation(self, terrain_type):
        """
        Sample precipitation for a node given its terrain.

        Args:
            terrain_type (str): Terrain label.

        Returns:
            float: Precipitation amount (clipped to [0, 50]).
        """
        params = self.precipitation_params[terrain_type]
        precipitation = (
            np.random.exponential(scale=params['scale'])
            if np.random.rand() < params['rain_probability']
            else 0.0
        )
        return np.clip(precipitation, 0, 50)

    def generate_temperature(self, terrain_type, time, precipitation):
        """
        Sample temperature for a node given terrain, local time, and precipitation.

        Args:
            terrain_type (str): Terrain label.
            time (int): Hour of day in [0, 23].
            precipitation (float): Precipitation amount.

        Returns:
            float: Temperature (clipped to [-10, 45]).
        """
        params = self.temp_params[terrain_type]
        # Diurnal sinusoid centered near 6:00; precipitation cools temperature.
        temp_variation = params['diurnal_variation'] * np.sin(((time - 6) / 24) * 2 * np.pi)
        temperature = params['base_temp'] + temp_variation - params['precipitation_effect'] * precipitation
        return np.clip(temperature + np.random.normal(0, 2), -10, 45)

    def generate_wind_speed(self, terrain_type, time, precipitation):
        """
        Sample wind speed for a node given terrain, local time, and precipitation.

        Args:
            terrain_type (str): Terrain label.
            time (int): Hour of day in [0, 23].
            precipitation (float): Precipitation amount.

        Returns:
            float: Wind speed (clipped to [0, 100]).
        """
        params = self.wind_params[terrain_type]
        wind_variation = params['time_variation'] * np.sin((time / 24) * 2 * np.pi)
        wind_speed = params['base_wind'] + wind_variation + params['precipitation_effect'] * precipitation
        return np.clip(wind_speed + np.random.normal(0, 2), 0, 100)

    def generate_visibility(self, terrain_type, time, precipitation):
        """
        Sample visibility for a node given terrain, local time, and precipitation.

        Args:
            terrain_type (str): Terrain label.
            time (int): Hour of day in [0, 23].
            precipitation (float): Precipitation amount.

        Returns:
            float: Visibility (clipped to [0, 100]).
        """
        params = self.visibility_params[terrain_type]
        visibility = params['max_visibility']
        # Night penalty: reduce visibility outside ~07:00–19:00, scaled by distance from midday.
        if time < 7 or time > 19:
            visibility -= params['time_effect'] * (abs(13 - time) / 6)
        visibility -= params['precipitation_effect'] * precipitation
        return np.clip(visibility + np.random.normal(0, 2), 0, 100)

    def generate_hexagonal_cell_network(self):
        """
        Build the hex grid and populate node metadata.

        Returns:
            tuple: (graph, pos, start_node, end_node) where
                graph (networkx.Graph): Graph with node attributes:
                    - terrain (str)
                    - time (int)
                    - temperature (float)
                    - wind_speed (float)
                    - visibility (float)
                    - precipitation (float)
                    - mine_presence (bool)
                    - neighbors (dict[int, Optional[str]])
                and edges with:
                    - weight (int)

                pos (dict[str, tuple[float, float]]): Node positions keyed by node id.
                start_node (str): Selected start node id.
                end_node (str): Selected end node id.
        """
        # Grid size (approx square)
        num_rows = int(np.sqrt(self.num_nodes))
        num_columns = num_rows
        hex_size = 1.0

        base_weight = 50

        graph = nx.Graph()
        pos = {}

        # Create nodes with axial coords, positions, and placeholders
        for row in range(num_rows):
            for column in range(num_columns):
                node = (row, column)
                x_pos = hex_size * np.sqrt(3) * (row + column / 2)
                y_pos = hex_size * (3 / 2) * column
                graph.add_node(node, row=row, column=column, pos=(x_pos, y_pos))
                pos[node] = (x_pos, y_pos)

        # Connect neighbors (edges are adjacency only)
        for row in range(num_rows):
            for column in range(num_columns):
                for row_delta, col_delta in self.DIRECTIONS:
                    n_row, n_col = row + row_delta, column + col_delta
                    if 0 <= n_row < num_rows and 0 <= n_col < num_columns:
                        graph.add_edge((row, column), (n_row, n_col), weight=50)

        # Time snapshot for env generation
        current_time = np.random.randint(0, 24)

        # Terrain assignment with local smoothing via transition matrix.
        # Row-major fill; for each node, look at already-assigned neighbors.
        for row in range(num_rows):
            for column in range(num_columns):
                node = (row, column)
                assigned_neighbors = []
                for row_delta, col_delta in self.DIRECTIONS:
                    n_row, n_col = row + row_delta, column + col_delta
                    if (n_row, n_col) in graph.nodes:
                        terrain = graph.nodes[(n_row, n_col)].get('terrain', None)
                        if terrain is not None:
                            assigned_neighbors.append(terrain)
                graph.nodes[node]['terrain'] = self.assign_terrain(assigned_neighbors)

        # Per-node environment + mines
        all_nodes = list(graph.nodes())
        num_mines = int(len(all_nodes) * self.mine_likelihood)
        mine_indices = set(np.random.choice(range(len(all_nodes)), num_mines, replace=False))

        for idx, node in enumerate(all_nodes):
            terrain = graph.nodes[node]['terrain']
            precipitation = self.generate_precipitation(terrain)
            temperature = self.generate_temperature(terrain, current_time, precipitation)
            wind_speed = self.generate_wind_speed(terrain, current_time, precipitation)
            visibility = self.generate_visibility(terrain, current_time, precipitation)
            mine_presence = (idx in mine_indices)

            graph.nodes[node].update({
                'time': current_time,
                'temperature': temperature,
                'wind_speed': wind_speed,
                'visibility': visibility,
                'precipitation': precipitation,
                'mine_presence': mine_presence,
                'weight': base_weight
            })

        # Build per-node neighbor map by direction index
        for row in range(num_rows):
            for column in range(num_columns):
                node = (row, column)
                neighbor_map = {}
                for dir_idx, (row_delta, col_delta) in enumerate(self.DIRECTIONS):
                    n_row, n_col = row + row_delta, column + col_delta
                    neighbor_map[dir_idx] = str((n_row, n_col)) if (n_row, n_col) in graph.nodes else None
                graph.nodes[node]['neighbors'] = neighbor_map

        # Choose interior start/end nodes
        interior = [
            node for node in graph.nodes()
            if 5 <= node[0] <= num_rows - 5 and 5 <= node[1] <= num_columns - 5
        ]
        if interior:
            start_node_tuple = min(interior, key=lambda x: (x[0], x[1]))
            end_node_tuple = max(interior, key=lambda x: (x[0], x[1]))
        else:
            start_node_tuple = all_nodes[0]
            end_node_tuple = all_nodes[-1]

        # Relabel nodes as strings for JSON/UI compatibility
        mapping = {node: str(node) for node in graph.nodes()}
        graph = nx.relabel_nodes(graph, mapping)
        pos = {mapping[node]: coord for node, coord in pos.items()}

        return graph, pos, str(start_node_tuple), str(end_node_tuple)

    def save_to_json(self, graph, start_node, end_node):
        """
        Serialize the graph into a JSON schema and write it to the output directory.

        Args:
            graph (networkx.Graph): Graph returned by generate_hexagonal_cell_network().
            start_node (str): Start node id.
            end_node (str): End node id.
        """
        network = {
            "mission": {
                "start": str(start_node),
                "end": str(end_node),
                "direction_order": {str(i): name for i, name in enumerate(self.DIR_NAMES)}
            },
            "nodes": [],
            "edges": []
        }

        # Nodes with metadata + neighbor pointers
        for node_id, attrs in graph.nodes(data=True):
            network["nodes"].append({
                "id": node_id,
                "metadata": {
                    "terrain": attrs["terrain"],
                    "time": attrs["time"],
                    "temperature": attrs["temperature"],
                    "wind_speed": attrs["wind_speed"],
                    "visibility": attrs["visibility"],
                    "precipitation": attrs["precipitation"],
                    "weight": attrs.get("weight", 50)
                },
                "neighbors": attrs.get("neighbors", {}),
                "inaccessible": {
                    "mine_presence": attrs["mine_presence"],
                }
            })

        # Edges (adjacency + weight only)
        for source, destination, edge_data in graph.edges(data=True):
            network["edges"].append({
                "from": source,
                "to": destination,
                "weight": edge_data.get("weight", 50)
            })

        out_path = os.path.join(self.output_dir, self.json_filename)
        with open(out_path, 'w') as file:
            json.dump(network, file, indent=4)

    def run(self):
        """
        Generate the hex grid, export JSON, and visualize the result.
        """
        graph, pos, start_node, end_node = self.generate_hexagonal_cell_network()
        self.save_to_json(graph, start_node, end_node)
