import pytest
import os
import copy
import networkx as nx
from operation_safe_passage.environment.environment import Environment


class TestEnvironment:

    TEST_NETWORK = {
        "mission": {
            "start": "(0, 0)",
            "end": "(1, 1)",
            "human estimate time": 30,
            "AI estimate time": 5,
            "UGV traversal time": 20,
            "UGV clear time": 60,
            "UAV traversal time": 1,
            "direction_order": {
                "0": "E",
                "1": "SE",
                "2": "SW",
                "3": "W",
                "4": "NW",
                "5": "NE"
            }
        },
        "nodes": [
            {
                "id": "(0, 0)",
                "metadata": {
                    'UGV_navigated': False,
                    'mine_detected': False,
                    "terrain": "Grassy",
                    "time": 23,
                    "temperature": 8.377090684800237,
                    "wind_speed": 7.244471327659357,
                    "visibility": 93.09058482349712,
                    "precipitation": 0.0
                },
                "neighbors": {
                    "0": "(1, 0)",
                    "1": "(0, 1)",
                    "2": None,
                    "3": None,
                    "4": None,
                    "5": None
                },
                "inaccessible": {
                    "mine_presence": False
                }
            },
            {
                "id": "(0, 1)",
                "metadata": {
                    'UGV_navigated': False,
                    'mine_detected': False,
                    "terrain": "Grassy",
                    "time": 23,
                    "temperature": 8.661341330168387,
                    "wind_speed": 7.102220321586332,
                    "visibility": 91.42327745658805,
                    "precipitation": 0.0
                },
                "neighbors": {
                    "0": "(1, 1)",
                    "1": None,
                    "2": None,
                    "3": None,
                    "4": "(0, 0)",
                    "5": "(1, 0)"
                },
                "inaccessible": {
                    "mine_presence": False
                }
            },
            {
                "id": "(1, 0)",
                "metadata": {
                    'UGV_navigated': False,
                    'mine_detected': False,
                    "terrain": "Grassy",
                    "time": 23,
                    "temperature": 10.209851412485316,
                    "wind_speed": 9.266940556851953,
                    "visibility": 92.24362669273417,
                    "precipitation": 0.0
                },
                "neighbors": {
                    "0": None,
                    "1": "(1, 1)",
                    "2": "(0, 1)",
                    "3": "(0, 0)",
                    "4": None,
                    "5": None
                },
                "inaccessible": {
                    "mine_presence": True
                }
            },
            {
                "id": "(1, 1)",
                "metadata": {
                    'UGV_navigated': False,
                    'mine_detected': False,
                    "terrain": "Grassy",
                    "time": 23,
                    "temperature": 6.611851723207108,
                    "wind_speed": 9.711177353894794,
                    "visibility": 28.968018363078155,
                    "precipitation": 30.462955987709186
                },
                "neighbors": {
                    "0": None,
                    "1": None,
                    "2": None,
                    "3": "(0, 1)",
                    "4": "(1, 0)",
                    "5": None
                },
                "inaccessible": {
                    "mine_presence": False
                }
            }
        ],
        "edges": [
            {
                "from": "(0, 0)",
                "to": "(1, 0)",
                "weight": 50
            },
            {
                "from": "(0, 0)",
                "to": "(0, 1)",
                "weight": 50
            },
            {
                "from": "(0, 1)",
                "to": "(1, 1)",
                "weight": 50
            },
            {
                "from": "(0, 1)",
                "to": "(1, 0)",
                "weight": 50
            },
            {
                "from": "(1, 0)",
                "to": "(1, 1)",
                "weight": 50
            }
        ]
    }

    @pytest.fixture
    def test_environment(self, mocker) -> Environment:
        """
        Function to test the Environments's init method

        Returns:
            Environments: the created environment
        """
        return Environment(nodes=self.TEST_NETWORK["nodes"], output_dir="test")

    def test_environment_reset(self, test_environment: Environment):
        """
        Function to test the Environments's reset method

        Args:
            test_environment: The created environment passed from the fixture
        """
        test_environment.network.nodes["(0, 0)"]["metadata"]["TEST"] = 1234
        test_environment.reset()
        assert "TEST" not in test_environment.network.nodes["(0, 0)"]["metadata"]

    def test_edge_iter_no_neighbors(self, test_environment: Environment):
        """
        Function to test the Environments's edge_data method when a node has no neighbors

        Args:
            test_environment: The created environment passed from the fixture
        """
        results = (
            test_environment._edge_iter(
                [
                    {
                        "id": "(1, 0)",
                        "mine_present": False,
                        "metadata": {
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 10.209851412485316,
                            "wind_speed": 9.266940556851953,
                            "visibility": 92.24362669273417,
                            "precipitation": 0.0
                        },
                        "neighbors": {
                            "0": None,
                            "1": "(1, 1)",
                            "2": None,
                            "3": None,
                            "4": None,
                            "5": None
                        },
                        "inaccessible": {
                            "mine_presence": False
                        }
                    },
                    {
                        "id": "(1, 1)",
                        "mine_present": False,
                        "metadata": {
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 6.611851723207108,
                            "wind_speed": 9.711177353894794,
                            "visibility": 28.968018363078155,
                            "precipitation": 30.462955987709186
                        },
                        "inaccessible": {
                            "mine_presence": False
                        }
                    }
                ]
                ,
                ["(1, 0)", "(1, 1)"],
                default_edge_weight=10
            )
        )
        assert len(list(results)) == 1

    def test_edge_data(self, test_environment: Environment):
        """
        Function to test the Environments's edge_data method

        Args:
            test_environment: The created environment passed from the fixture
        """
        assert "{'weight': 50}" == str(test_environment.edge_data("(0, 0)", "(0, 1)"))    
        assert "None" == str(test_environment.edge_data("0", "3"))
    
    def test_node_data(self, test_environment: Environment):
        """
        Function to test the Environments's node_data method

        Args:
            test_environment: The created environment passed from the fixture
        """
        assert "{'metadata': {'UGV_navigated': False, 'mine_detected': False, 'terrain': 'Grassy', 'time': 23, 'temperature': 8.377090684800237, 'wind_speed': 7.244471327659357, 'visibility': 93.09058482349712, 'precipitation': 0.0}, 'neighbors': {'0': '(1, 0)', '1': '(0, 1)', '2': None, '3': None, '4': None, '5': None}, 'inaccessible': {'mine_presence': False}}" == str(test_environment.node_data("(0, 0)"))
    
    def test_has_mine(self, test_environment: Environment):
        """
        Function to test the Environments's edge_data method

        Args:
            test_environment: The created environment passed from the fixture
        """
        assert False == test_environment.has_mine("(0, 0)")
        assert True == test_environment.has_mine("(1, 0)")

    def test_set_node_weight(self, test_environment: Environment):
        """
        Function to test the Environments's set_node_weight method

        Args:
            test_environment: The created environment passed from the fixture
        """
        node_id = "(0, 0)"
        test_environment.set_node_weight(node_id, 99)
        assert 99 == test_environment.network.nodes[node_id]["weight"]

    def test_edge_w(self, test_environment: Environment):
        """
        Tests the Environments's edge_w method

        Args:
            test_environment (Environment): The Environment object returned from the fixture
        """
        assert 50 == test_environment.edge_w("(0, 0)", "(0, 1)", {})

    def test_plan_path(self, test_environment: Environment):
        """
        Tests the Environments's plan_path method

        Args:
            test_environment (Environment): The Environment object returned from the fixture
        """
        assert ["(0, 0)", "(0, 1)", "(1, 1)"] == test_environment.plan_path("(0, 0)", "(1, 1)")

    def test_visualize_network(self, test_environment: Environment, mocker):
        """
        Tests the Environments's visualize_network method

        Args:
            test_environment (Environment): The Environment object returned from the fixture
            mocker (pytest.Mocker): An object for creating Mock objects
        """
        mocker.patch("operation_safe_passage.environment.environment.plt.figure")
        mocker.patch("operation_safe_passage.environment.environment.os.path.dirname", return_value="/test")
        mock_savefig = mocker.patch("operation_safe_passage.environment.environment.plt.savefig")
        test_environment.visualize_network(start_node="(0, 0)", end_node="(1, 1)", show_labels=True)
        mock_savefig.assert_called_with(os.path.join("test", "generated_map.png"), bbox_inches='tight', dpi=200)

    def test_visualize_network_other(self, mocker):
        """
        Tests conditionals of the Environment's visualize_network method

        Args:
            mocker (pytest.Mocker): The tool used to create Mock objects
        """
        new_network = copy.deepcopy(self.TEST_NETWORK)
        for node in new_network["nodes"]:
            node["id"] = node["id"].replace(",", ";")

        def mocked_sping_loadout(network, seed):
            raise ValueError
        mocker.patch("operation_safe_passage.environment.environment.nx.spring_layout", side_effect=mocked_sping_loadout)
        with pytest.raises(ValueError): 
            new_environment = Environment(new_network["nodes"])
            new_environment.visualize_network(start_node="(0, 0)", end_node="(1, 1)")

    def test_mission_parse_node_id(self, test_environment: Environment):
        """
        Tests the Environment's _parse_node_id method

        Args:
            test_environment (Environment): The Environment object to test
        """
        first, second = test_environment._parse_node_id(" ( 0 , 1 )")
        assert first == 0 and second == 1

    def test_mission_control_hex_distance(self, test_environment: Environment):
        """
        Tests the Environment's _hex_distance method

        Args:
            test_environment (Environment): The Environment object to test
        """
        assert 2 == test_environment._hex_distance("(0, 0)", "1, 1")

    def test_mission_weighted_distance(self, test_environment: Environment):
        """
        Tests the Environment's _weighted_distance method

        Args:
            test_environment (Environment): The Environment object to test
        """
        assert -1 == test_environment._weighted_distance(None, "(1, 1)", default=-1)
        assert -1 == test_environment._weighted_distance("(0, 0)", None, default=-1)
        with pytest.raises(nx.exception.NodeNotFound):
            assert -1 == test_environment._weighted_distance("(0, 0)", "(2, 2)", default=-1)
        assert 3 == test_environment._weighted_distance("(0, 0)", "(1, 1)", default=-1)
