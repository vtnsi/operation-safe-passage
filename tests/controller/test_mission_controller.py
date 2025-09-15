import pytest
from dataclasses import dataclass
from operation_safe_passage.controller.mission_controller import MissionController


class TestMissionController():

    TEST_NETWORK = {
        "mission": {
            "start": "(0, 0)",
            "end": "(1, 1)",
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
                'UGV_navigated': False,
                'mine_detected': False,
                "metadata": {
                    "terrain": "Grassy",
                    "time": 23,
                    "temperature": 10.209851412485316,
                    "wind_speed": 9.266940556851953,
                    "visibility": 92.24362669273417,
                    "precipitation": 0.0,
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

    TEST_PARAMS = {
        "num_nodes" : 4,
        "mine_likelihood" : 0.2,

        "processing_params": {
            "UGV_traversal_time": 20,
            "UGV_clear_time": 60,
            "UAV_traversal_time": 1
        },
        "terrain_coeff": {"Grassy": 1.0, "Rocky": 0.5, "Sandy": 0.0, "Wooded": -0.25, "Swampy": -0.75},
        "scanner_params": {
            "scanners": {
                "default": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":1},
                "degraded": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":1},
                "baseline": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":30}
            }
        },
        "agents": {
            "uavs": [
                { "name": "uav", "scanners": ["default", "degraded"] },
            ],
            "num_ugvs": 1
        }
    }

    class MissionControllerSideEffect:
        def __init__(self, *functions):
            self.functions = iter(functions)

        def __call__(self, *args, **kwargs):
            function = next(self.functions)
            return function(*args, **kwargs)

    @pytest.fixture
    def test_mission_controller(self, mocker) -> MissionController:
        """
        Tests the MissionControler __init__ method

        Returns
            (MissionController): The created MissionController object
        """
        mocked_open = mocker.mock_open(read_data="")
        mocker.patch("builtins.open", mocked_open)

        def mocked_load_network(_filename):
            return self.TEST_NETWORK

        def mocked_load_params(_filename):
            return self.TEST_PARAMS
        mocker.patch("operation_safe_passage.controller.validate_mission_params.JsonValidator.validate_config")
        mocker.patch("json.load", side_effect=self.MissionControllerSideEffect(mocked_load_network, mocked_load_params))
        mocker.patch("os.makedirs")
        mission_controller = MissionController(param_path="test/params", network_path="test/network", output_dir="test/output")
        assert mission_controller.start_node == "(0, 0)"
        assert mission_controller.end_node == "(1, 1)"
        return mission_controller        

    def test_mission_control_reset(self, test_mission_controller: MissionController):
        """
        Tests the MissionController's reset method

        Args:
            test_mission_controller (MissionController): The MissionController object to test
        """
        actions = {
            test_mission_controller.agents[0]: {
                "move": "NE",
                "scan": "default"
            },
            test_mission_controller.agents[1]: {
                "move": "E"
            }
        }
        expected_node_dict = {
            "id": "(1, 0)",
            "inaccessible": {
                "mine_presence": True,
            },
            "metadata": {
                'UGV_navigated': True,
                'mine_detected': True,
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
            }
        }
        test_mission_controller.step(actions)
        test_mission_controller.reset()
        assert test_mission_controller.time.current_time == 0
        assert test_mission_controller.agents[0].current_node == "(0, 0)"
        assert test_mission_controller.agents[1].current_node == "(0, 0)"
        assert expected_node_dict == test_mission_controller._node_dict("(1, 0)")

    def test_mission_control_get_state(self, test_mission_controller: MissionController):
        """
        Tests the MissionController's get_state method

        Args:
            test_mission_controller (MissionController): The MissionController object to test
        """
        expected_state = {
            "time": 0.0,
            "agents": [
                {
                    "type": "uav",
                    "current": {
                        'UGV_navigated': False,
                        'mine_detected': False,
                        "terrain": "Grassy",
                        "time": 23,
                        "temperature": 8.377090684800237,
                        "wind_speed": 7.244471327659357,
                        "visibility": 93.09058482349712,
                        "precipitation": 0.0,
                    },
                    "neighbors": {
                        "E": {
                            'UGV_navigated': False,
                            'mine_detected': False,
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 10.209851412485316,
                            "wind_speed": 9.266940556851953,
                            "visibility": 92.24362669273417,
                            "precipitation": 0.0,
                        },
                        "NE": {
                            'UGV_navigated': False,
                            'mine_detected': False,
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 8.661341330168387,
                            "wind_speed": 7.102220321586332,
                            "visibility": 91.42327745658805,
                            "precipitation": 0.0
                        }
                    },
                    "distance_to_goal": 2.0,
                    "distance_to_goal_weighted": 3.0,
                    "distance_to_agents": {
                        "UGV_0": 0.0
                    },
                    "previous_node": None,
                    "previous_action": None,
                    "num_moves": 0,
                },
                {
                    "type": "UGV_0",
                    "current": {
                        'UGV_navigated': False,
                        'mine_detected': False,
                        "terrain": "Grassy",
                        "time": 23,
                        "temperature": 8.377090684800237,
                        "wind_speed": 7.244471327659357,
                        "visibility": 93.09058482349712,
                        "precipitation": 0.0,
                    },
                    "neighbors": {
                        "E": {
                            'UGV_navigated': False,
                            'mine_detected': False,
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 10.209851412485316,
                            "wind_speed": 9.266940556851953,
                            "visibility": 92.24362669273417,
                            "precipitation": 0.0,
                        },
                        "NE": {
                            'UGV_navigated': False,
                            'mine_detected': False,
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 8.661341330168387,
                            "wind_speed": 7.102220321586332,
                            "visibility": 91.42327745658805,
                            "precipitation": 0.0
                        }
                    },
                    "distance_to_goal": 2.0,
                    "distance_to_goal_weighted": 3.0,
                    "distance_to_agents": {
                        "uav": 0.0
                    },
                    "previous_node": None,
                    "previous_action": None,
                    "num_moves": 0,
                }
            ]
        }
        assert expected_state == test_mission_controller.get_state()

    def test_dir_from_step(self, test_mission_controller: MissionController):
        """
        Tests the MissionController's dir_from_step method

        Args:
            test_mission_controller (MissionController): The MissionController object to test
        """
        assert "NE" == test_mission_controller.dir_from_step("(0, 0)", "(0, 1)")
        with pytest.raises(RuntimeError):
            test_mission_controller.dir_from_step("(0, 0)", "(1, 1)")

    def test_mission_control_step(self, test_mission_controller: MissionController):
        """
        Tests the MissionController's step method

        Args:
            test_mission_controller (MissionController): The MissionController object to test
        """
        actions = {
            test_mission_controller.agents[0]: {
                "move": "NE",
                "scan": "default"
            },
            test_mission_controller.agents[1]: {
                "move": "E"
            }
        }
        test_mission_controller.step(actions)
        assert test_mission_controller.time.get_current_time() == 82

    def test_mission_control_public_node_view(self, test_mission_controller: MissionController):
        """
        Tests the MissionController's public_node_view method

        Args:
            test_mission_controller (MissionController): The MissionController object to test
        """
        expected_node_view = {
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
            }
        }
        assert expected_node_view == test_mission_controller._public_node_view(self.TEST_NETWORK["nodes"][0])

    @dataclass
    class MoveAgentDataClass:
        """
        A dataclass to help test the _resolve_scanner_name test cases
        """
        direction: str
        expected_result_bool: bool
        expected_result_id: dict
        id: str

    @pytest.mark.parametrize(
        "test_case",
        [
            MoveAgentDataClass("noop", False, "(0, 0)", "test_mission_control_move_agent_and_get_node_noop"),
            MoveAgentDataClass(0, True, "(1, 0)", "test_mission_control_move_agent_and_get_node_normal_int"),
            MoveAgentDataClass("E", True, "(1, 0)", "test_mission_control_move_agent_and_get_node_normal_str"),
            MoveAgentDataClass(8, False, "(0, 0)", "test_mission_control_move_agent_and_get_node_normal_str"),
            MoveAgentDataClass("NOEXIST", False, None, "test_mission_control_move_agent_and_get_node_oob"),
            MoveAgentDataClass(2, False, "(0, 0)", "test_mission_control_move_agent_and_get_node_blocked"),
        ],
        ids=lambda tc: tc.id
    )
    def test_mission_control_move_agent_and_get_node(self, test_mission_controller: MissionController, test_case: MoveAgentDataClass):
        """
        Tests the MissionController's move_agent_and_get_node method

        Args:
            test_mission_controller (MissionController): The MissionController object to test
        """
        passed, returned_dict = test_mission_controller._move_agent_and_get_node(test_mission_controller.agents[0], test_case.direction)
        assert test_case.expected_result_bool == passed
        if test_case.expected_result_id == None:
            assert returned_dict == None
        else:
            assert test_case.expected_result_id == returned_dict["id"]

    def test_mission_control_node_dict(self, test_mission_controller: MissionController):
        """
        Tests the MissionController's _node_dict method

        Args:
            test_mission_controller (MissionController): The MissionController object to test
        """
        expected_node_view = {
            "id": "(0, 0)",
            "inaccessible": {
                "mine_presence": False,
            },
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
            }
        }
        assert expected_node_view == test_mission_controller._node_dict("(0, 0)")
