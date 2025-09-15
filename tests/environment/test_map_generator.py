import pytest
import os
import numpy as np
from unittest.mock import patch
from dataclasses import dataclass
from operation_safe_passage.environment.map_generator import MapGenerator


class TestMapGenerator:

    TEST_PARAMS = {
        "environment": {
            "num_nodes" : 5,
            "mine_likelihood" : 0.2,

            "terrain_types" : ["Grassy", "Sandy"],
            "transition_matrix" : [
                    [0.5, 0.125], 
                    [0.05, 0.6]
                ],

            "precipitation_params" : {
                    "Grassy": {
                        "rain_probability": 0.1,
                        "scale": 10
                    },
                    "Sandy": {
                        "rain_probability": 0.02,
                        "scale": 4
                    }
                },

            "temp_params" : {
                    "Grassy": {"base_temp": 20, "diurnal_variation": 10, "precipitation_effect": 0.1},
                    "Sandy": {"base_temp": 25, "diurnal_variation": 12, "precipitation_effect": 0.1}
                },

            "wind_params" : {
                    "Grassy": {"base_wind": 10, "time_variation": 5, "precipitation_effect": 0.05},
                    "Sandy": {"base_wind": 12, "time_variation": 6, "precipitation_effect": 0.3}
                },

            "visibility_params" : { 
                    "Grassy": {"max_visibility": 100, "time_effect": 5, "precipitation_effect": 2},
                    "Sandy": {"max_visibility": 100, "time_effect": 4, "precipitation_effect": 1.5}
                }
        }
    }

    @pytest.fixture
    def test_map_generator(self, mocker) -> MapGenerator:
        """
        Function to test the MapGenerator's init method

        Returns:
            MapGenerator: the created map generator
        """
        mocked_open = mocker.mock_open(read_data="")
        mocker.patch("builtins.open", mocked_open)

        def mocked_load(filename):
            return self.TEST_PARAMS
        mocker.patch("json.load", side_effect=mocked_load)
        return MapGenerator(os.path.join("tests", "configs", "test_params.json"))

    @dataclass
    class MapGeneratorTestCase:
        # Dont start helper class names with "Test"
        terrain_input: str
        expected_result: tuple
        id: str
        num_epochs: int = 1000
        time: int = None
        precipitation: int = None

    def __monte_carlo_generation(self, test_case: MapGeneratorTestCase, function_name) -> bool:
        """
        Runs a Monte Carlo simulation to determine the average value of a given test case

        Args:
            test_case (MapGeneratorTestCase): The test case to extract values from
            function_name (function): The name of the function to run

        Returns:
            bool: True if the test results are expected given the test case information
        """
        average_value = 0
        for _i in range(0, test_case.num_epochs):
            if test_case.time != None:
                average_value += function_name(test_case.terrain_input, test_case.time, test_case.precipitation)
            else:
                average_value += function_name(test_case.terrain_input)
        average_value /= test_case.num_epochs
        return test_case.expected_result[0] <= average_value <= test_case.expected_result[1]

    @pytest.mark.parametrize(
        "assign_terrain_test_case",
        [
            MapGeneratorTestCase([], (0.35, 0.65), "test_random_terrain"),
            MapGeneratorTestCase(["Grassy", "Grassy", "Grassy", "Grassy", "Grassy", "Grassy"], (0.7, 0.9), "test_all_grassy"),
            MapGeneratorTestCase(["Sandy", "Sandy", "Sandy", "Sandy", "Sandy", "Sandy"], (0.0, 0.15), "test_all_sandy"),
            MapGeneratorTestCase(["Grassy", "Sandy", "Grassy", "Sandy", "Grassy", "Sandy"], (0.3, 0.55), "test_half_grassy")
        ],
        ids=lambda tc: tc.id
    )
    def test_assign_terrain(self, test_map_generator: MapGenerator, assign_terrain_test_case: MapGeneratorTestCase):
        """
        Tests the MapGenerator's assign_terrain function

        Args:
            test_map_generator (MapGenerator): The MapGenerator object returned from the fixture
            assign_terrain_test_case (MapGeneratorTestCase): The information for the current test case
        """
        count_grassy = 0
        for _i in range(0, assign_terrain_test_case.num_epochs):
            terrain = test_map_generator.assign_terrain(assign_terrain_test_case.terrain_input)
            if terrain == "Grassy":
                count_grassy += 1
        assert assign_terrain_test_case.expected_result[0] <= count_grassy / assign_terrain_test_case.num_epochs <= assign_terrain_test_case.expected_result[-1]

    @pytest.mark.parametrize(
        "generate_precipitation_test_case",
        [
            MapGeneratorTestCase("Grassy", (0, 2.5), "test_generate_precipitation_grassy"),
            MapGeneratorTestCase("Sandy", (0, 1.2), "test_generate_precipitation_sandy")
        ],
        ids=lambda tc: tc.id
    )
    def test_generate_precipitation(self, test_map_generator: MapGenerator, generate_precipitation_test_case: MapGeneratorTestCase):
        """
        Tests the MapGenerator's generate_precipitation function

        Args:
            test_map_generator (MapGenerator): The MapGenerator object returned from the fixture
            generate_precipitation_test_case (MapGeneratorTestCase): The information for the current test case
        """
        assert self.__monte_carlo_generation(generate_precipitation_test_case, test_map_generator.generate_precipitation)

    @pytest.mark.parametrize(
        "generate_temperature_test_case",
        [
            MapGeneratorTestCase("Grassy", (29, 31), "test_generate_temperature_noon", time=12, precipitation=0),
            MapGeneratorTestCase("Grassy", (26.5, 28.5), "test_generate_temperature_noon_with_high_precipitation", time=12, precipitation=25),
            MapGeneratorTestCase("Grassy", (28, 30), "test_generate_temperature_noon_with_high_precipitation", time=12, precipitation=10),
            MapGeneratorTestCase("Grassy", (19, 21), "test_generate_temperature_dawn", time=6, precipitation=0),
            MapGeneratorTestCase("Grassy", (19, 21), "test_generate_temperature_dusk", time=18, precipitation=0),
        ],
        ids=lambda tc: tc.id
    )
    def test_generate_temperature(self, test_map_generator: MapGenerator, generate_temperature_test_case: MapGeneratorTestCase):
        """
        Tests the MapGenerator's generate_temperature method

        Args:
            test_map_generator (MapGenerator): The MapGenerator object returned from the fixture
            generate_temperature_test_case (MapGeneratorTestCase): The information for the current test case 
        """
        assert self.__monte_carlo_generation(generate_temperature_test_case, test_map_generator.generate_temperature)


    @pytest.mark.parametrize(
        "generate_wind_speed_test_case",
        [
            MapGeneratorTestCase("Grassy", (9, 11), "test_wind_speed_noon", time=12, precipitation=0),
            MapGeneratorTestCase("Grassy", (10, 12), "test_wind_speed_noon_with_high_precipitation", time=12, precipitation=25),
            MapGeneratorTestCase("Grassy", (9.5, 11.5), "test_wind_speed_noon_with_high_precipitation", time=12, precipitation=10),
            MapGeneratorTestCase("Grassy", (14, 16), "test_wind_speed_noon", time=6, precipitation=0),
            MapGeneratorTestCase("Grassy", (4, 6), "test_wind_speed_noon", time=18, precipitation=0),
        ],
        ids=lambda tc: tc.id
    )
    def test_generate_wind_speed(self, test_map_generator: MapGenerator, generate_wind_speed_test_case: MapGeneratorTestCase):
        """
        Tests the MapGenerator's generate_wind_speed method

        Args:
            test_map_generator (MapGenerator): The MapGenerator object returned from the fixture
            generate_wind_speed_test_case (MapGeneratorTestCase): The information for the current test case 
        """
        assert self.__monte_carlo_generation(generate_wind_speed_test_case, test_map_generator.generate_wind_speed)

    @pytest.mark.parametrize(
        "generate_visibility_test_case",
        [
            MapGeneratorTestCase("Grassy", (98, 100), "test_visibility_noon", time=12, precipitation=0),
            MapGeneratorTestCase("Grassy", (49, 51), "test_visibility_noon_with_high_precipitation", time=12, precipitation=25),
            MapGeneratorTestCase("Grassy", (79, 81), "test_visibility_noon_with_high_precipitation", time=12, precipitation=10),
            MapGeneratorTestCase("Grassy", (93, 95), "test_visibility_noon", time=6, precipitation=0),
            MapGeneratorTestCase("Grassy", (98, 100), "test_visibility_noon", time=18, precipitation=0),
        ],
        ids=lambda tc: tc.id
    )
    def test__generate_visibility(self, test_map_generator: MapGenerator, generate_visibility_test_case: MapGeneratorTestCase):
        """
        Tests the MapGenerator's generate_visibility method

        Args:
            test_map_generator (MapGenerator): The MapGenerator object returned from the fixture
            generate_visibility_test_case (MapGeneratorTestCase): The information for the current test case 
        """
        assert self.__monte_carlo_generation(generate_visibility_test_case, test_map_generator.generate_visibility)

    @pytest.fixture
    def test_generate_hexagonal_cell_network(self, test_map_generator: MapGenerator):
        """
        Tests the MapGenerator's generate_hexagonal_cell_network method

        Args:
            test_map_generator (MapGenerator): The MapGenerator object returned from the fixture
        
        Returns:
            The created hexagonal terrain map
            The positional information for the graph
            The start_node of the mission
            The end_node of the mission
        """
        test_map_generator.num_nodes = 121
        graph, _positions, start_node, end_node = test_map_generator.generate_hexagonal_cell_network()
        assert start_node == "(5, 5)"
        assert end_node == "(6, 6)"

        test_map_generator.num_nodes = 4
        graph, _positions, start_node, end_node = test_map_generator.generate_hexagonal_cell_network()
        assert start_node == "(0, 0)"
        assert end_node == "(1, 1)"
        assert len(list(graph.nodes())) == 4  # Nearest square less than 500 as set in test_params
        for node in list(graph.nodes()):
            assert graph.nodes[node]["terrain"] == "Grassy" or graph.nodes[node]["terrain"] == "Sandy"
            assert graph.nodes[node]["time"] != None
            assert graph.nodes[node]["temperature"] != None
            assert graph.nodes[node]["wind_speed"] != None
            assert graph.nodes[node]["visibility"] != None
            assert graph.nodes[node]["precipitation"] != None
            assert graph.nodes[node]["mine_presence"] != None
        return (graph, _positions, start_node, end_node)

    def test_save_to_json(self, test_map_generator: MapGenerator, test_generate_hexagonal_cell_network, mocker):
        """
        Tests the MapGenerator's save_to_json method

        Args:
            test_map_generator (MapGenerator): The MapGenerator object returned from the fixture
            test_generate_hexagonal_cell_network (networkx.Graph, dict, str, str):
                The created hexagonal terrain map
                The positional information for the graph
                The start_node of the mission
                The end_node of the mission
            mocker (pytest.Mocker): An object for creating Mock objects
        """
        mocked_open = mocker.mock_open()
        mocker.patch("builtins.open", mocked_open)

        def mocked_dump(graph_data, filename, indent):
            assert indent == 4
            for i in range(0, len(graph_data["nodes"])):
                graph_data["nodes"][i]["metadata"]["precipitation"] = np.float64(0.0)
                graph_data["nodes"][i]["metadata"]["temperature"] = np.float64(0.0)
                graph_data["nodes"][i]["metadata"]["terrain"] = np.str_("Grassy")
                graph_data["nodes"][i]["metadata"]["time"] = 0
                graph_data["nodes"][i]["metadata"]["visibility"] = np.float64(0.0)
                graph_data["nodes"][i]["metadata"]["wind_speed"] = np.float64(0.0)
            assert graph_data == {
                'edges': [
                    {'from': '(0, 0)', 'to': '(1, 0)', 'weight': 50},
                    {'from': '(0, 0)', 'to': '(0, 1)', 'weight': 50},
                    {'from': '(0, 1)', 'to': '(1, 1)', 'weight': 50},
                    {'from': '(0, 1)', 'to': '(1, 0)', 'weight': 50},
                    {'from': '(1, 0)', 'to': '(1, 1)', 'weight': 50},
                ],
                'mission': {
                    "start": "(0, 0)",
                    "end": "(1, 1)",
                    "direction_order": {
                        "0": "E",
                        "1": "NE",
                        "2": "NW",
                        "3": "W",
                        "4": "SW",
                        "5": "SE"
                    }
                },
                'nodes': [{
                    'id': '(0, 0)',
                    'inaccessible': {
                        'mine_presence': False,
                    },
                    'metadata': {
                        'precipitation': np.float64(0.0),
                        'temperature': np.float64(0.0),
                        'terrain': np.str_('Grassy'),
                        'time': 0,
                        'visibility': np.float64(0.0),
                        'weight': 50,
                        'wind_speed': np.float64(0.0),
                    },
                    'neighbors': {
                        0: '(1, 0)',
                        1: '(0, 1)',
                        2: None,
                        3: None,
                        4: None,
                        5: None,
                    },
                },
                {
                    'id': '(0, 1)',
                    'inaccessible': {
                        'mine_presence': False,
                    },
                    'metadata': {
                        'precipitation': np.float64(0.0),
                        'temperature': np.float64(0.0),
                        'terrain': np.str_('Grassy'),
                        'time': 0,
                        'visibility': np.float64(0.0),
                        'weight': 50,
                        'wind_speed': np.float64(0.0),
                    },
                    'neighbors': {
                        0: '(1, 1)',
                        1: None,
                        2: None,
                        3: None,
                        4: '(0, 0)',
                        5: '(1, 0)',
                    },
                },
                {
                    'id': '(1, 0)',
                    'inaccessible': {
                        'mine_presence': False,
                    },
                    'metadata': {
                        'precipitation': np.float64(0.0),
                        'temperature': np.float64(0.0),
                        'terrain': np.str_('Grassy'),
                        'time': 0,
                        'visibility': np.float64(0.0),
                        'weight': 50,
                        'wind_speed': np.float64(0.0),
                    },
                    'neighbors': {
                        0: None,
                        1: '(1, 1)',
                        2: '(0, 1)',
                        3: '(0, 0)',
                        4: None,
                        5: None,
                    },
                },
                {
                    'id': '(1, 1)',
                    'inaccessible': {
                        'mine_presence': False,
                    },
                    'metadata': {
                        'precipitation': np.float64(0.0),
                        'temperature': np.float64(0.0),
                        'terrain': np.str_('Grassy'),
                        'time': 0,
                        'visibility': np.float64(0.0),
                        'weight': 50,
                        'wind_speed': np.float64(0.0),
                    },
                    'neighbors': {
                        0: None,
                        1: None,
                        2: None,
                        3: '(0, 1)',
                        4: '(1, 0)',
                        5: None,
                    },
                }]
            }
        mocker.patch("json.dump", side_effect=mocked_dump)
        test_map_generator.save_to_json(
            test_generate_hexagonal_cell_network[0],
            test_generate_hexagonal_cell_network[2],
            test_generate_hexagonal_cell_network[3]
        )

    @patch.object(MapGenerator, "generate_hexagonal_cell_network", return_value=(0,0,0,0))
    @patch.object(MapGenerator, "save_to_json")
    def test_run(self, mock_generate_hexagonal_cell_network, mock_save_to_json, test_map_generator: MapGenerator):
        """
        Tests the MapGenerator's run method

        Args:
            mock_generate_hexagonal_cell_network (Mock): A blank replacement for the MapGenerator.generate_hexagonal_cell_network method
            mock_save_to_json (Mock): A blank replacement for the MapGenerator.save_to_json method
            test_map_generator (MapGenerator): The MapGenerator object returned from the fixture
        """
        test_map_generator.run()
        mock_generate_hexagonal_cell_network.assert_called()
        mock_save_to_json.assert_called()
