import numpy as np
import pytest
from unittest.mock import patch
from dataclasses import dataclass
from operation_safe_passage.controller.osp_gym import OSPGym


class TestOSPGym:

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
                    "precipitation": 0.0,
                    "weight": 50
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
                    "precipitation": 0.0,
                    "weight": 50
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
                    "precipitation": 0.0,
                    "weight": 50
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
                    "precipitation": 30.462955987709186,
                    "weight": 50
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
    def test_osp_gym(self, mocker):
        """
        Tests the OSPGym object's init method
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
        gym = OSPGym()
        return gym

    @patch.object(OSPGym, "_actions_to_controller_format")
    def test_gym_reset(self, mock_actions_to_controller_format, test_osp_gym: OSPGym):
        """
        Tests the OSPGym's reset method

        Args:
            test_osp_gym (OSPGym): The OSPGym object from the fixture
        """
        actions = {
            test_osp_gym.controller.agents[0]: {
                "move": "NE",
                "scan": "default"
            },
            test_osp_gym.controller.agents[1]: {
                "move": "E"
            }
        }
        mock_actions_to_controller_format.return_value = actions
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
                        'weight': 50
                    },
                    "neighbors": {
                        "E": {
                            'UGV_navigated': True,
                            'mine_detected': True,
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 10.209851412485316,
                            "wind_speed": 9.266940556851953,
                            "visibility": 92.24362669273417,
                            "precipitation": 0.0,
                            'weight': 50
                        },
                        "NE": {
                            'UGV_navigated': False,
                            'mine_detected': False,
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 8.661341330168387,
                            "wind_speed": 7.102220321586332,
                            "visibility": 91.42327745658805,
                            "precipitation": 0.0,
                            'weight': 50
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
                        'weight': 50
                    },
                    "neighbors": {
                        "E": {
                            'UGV_navigated': True,
                            'mine_detected': True,
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 10.209851412485316,
                            "wind_speed": 9.266940556851953,
                            "visibility": 92.24362669273417,
                            "precipitation": 0.0,
                            'weight': 50
                        },
                        "NE": {
                            'UGV_navigated': False,
                            'mine_detected': False,
                            "terrain": "Grassy",
                            "time": 23,
                            "temperature": 8.661341330168387,
                            "wind_speed": 7.102220321586332,
                            "visibility": 91.42327745658805,
                            "precipitation": 0.0,
                            'weight': 50
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
        test_osp_gym.step(actions)
        test_osp_gym.reset(seed=42)
        assert 0 == test_osp_gym._steps
        state = test_osp_gym.controller.get_state()
        state["agents"][0]["neighbors"]["NE"]["weight"] = 50
        state["agents"][1]["neighbors"]["NE"]["weight"] = 50
        assert expected_state == test_osp_gym.controller.get_state()

    @patch.object(OSPGym, "_actions_to_controller_format")  
    def test_gym_step(self, mock_actions_to_controller_format, test_osp_gym: OSPGym):
        """
        Tests the OSPGym's step method

        Args:
            test_osp_gym (OSPGym): The OSPGym object from the fixture
        """
        actions_1 = {
            test_osp_gym.controller.agents[0]: {
                "move": "NE",
                "scan": "default"
            },
            test_osp_gym.controller.agents[1]: {
                "move": "NE"
            }
        }
        actions_2 = {
            test_osp_gym.agents[0]: {
                "move": "noop"
            },
            test_osp_gym.agents[1]: {
                "move": "E"
            }
        }
        expected_result = [1]
        mock_actions_to_controller_format.return_value = actions_1
        obs, reward, terminated, truncated, _info = test_osp_gym.step(actions_1)
        assert terminated == False
        assert truncated == False
        assert reward == -1
        mock_actions_to_controller_format.return_value = actions_2
        obs, reward, terminated, truncated, _info = test_osp_gym.step(actions_2)
        assert test_osp_gym._steps == 2
        assert reward == 999.0
        assert terminated == True
        assert truncated == False

    def test_gym_render(self, test_osp_gym: OSPGym):
        """
        Tests the OSPGym's render method

        Args:
            test_osp_gym (OSPGym): The OSPGym object from the fixture
        """
        expected_result =  "time=0.0\nuav: d_goal=2.0, w=50.00\nUGV_0: d_goal=2.0, w=50.00"
        assert expected_result == test_osp_gym.render()

    def test_gym_ugv_reached_goal(self, test_osp_gym: OSPGym):
        """
        Tests the OSPGym's _ugv_reached_goal method

        Args:
            test_osp_gym (OSPGym): The OSPGym object from the fixture
        """
        test_state = {
            "agents": [
                {"type": "UGV", "distance_to_goal": 1.0}
            ]
        }
        assert False == test_osp_gym._ugv_reached_goal(test_state)
        test_state["agents"][0]["distance_to_goal"] = 0.0
        assert True == test_osp_gym._ugv_reached_goal(test_state)

    def test_gym_default_reward(self, test_osp_gym: OSPGym):
        """
        Tests the OSPGym's _default_reward method

        Args:
            test_osp_gym (OSPGym): The OSPGym object from the fixture
        """
        assert 989 == test_osp_gym._default_reward({}, True, True)

    @dataclass
    class ActionsToControllerFormatTestCase:
        """
        A dataclass to help test the _resolve_scanner_name test cases
        """
        agent: str
        action: dict
        expected_result: dict
        id: str
        expected_error: type[Exception] = None
        error_message: str | None = None

    def test_gym_actions_to_controller_format(self, test_osp_gym: OSPGym):
        """
        Tests the OSPGym's _actions_to_controller_format method

        Args:
            test_osp_gym (OSPGym): The OSPGym object from the fixture
        """
        action = [1, 0, 6]
        action_too_long = [1, 0, 6, 1, 1, 1]
        action_oob = [1, 0, 99]
        expected_result = {
            test_osp_gym.controller.agents[0]: {
                "move": "NE",
                "scan": "default"
            },
            test_osp_gym.controller.agents[1]: {
                "move": None,
                "scan": None
            }
        }
        assert expected_result == test_osp_gym._actions_to_controller_format(action)
        with pytest.raises(ValueError, match="Flat action length 6 != expected 3"):
            test_osp_gym._actions_to_controller_format(action_too_long)
        with pytest.raises(ValueError, match="Move index out of range: 99"):
            test_osp_gym._actions_to_controller_format(action_oob)

    def test_gym_build_obs_array(self, test_osp_gym: OSPGym):
        """
        Tests the OSPGym's _build_obs_array method

        Args:
            test_osp_gym (OSPGym): The OSPGym object from the fixture
        """
        blocks: list[np.ndarray] = []
        block_1 = np.array(
            [
                0.0,
                2.0,
                3.0,
                50.0,
                -1.0,
                8.377090684800237,
                7.244471327659357,
                93.09058482349712,
                0.0,
                *[50.0, 50.0, -1.0, -1.0, -1.0, -1.0],
                *[0.0],
            ],
            dtype=np.float32,
        )
        block_2 = np.array(
            [
                0.0,
                2.0,
                3.0,
                50.0,
                -1.0,
                8.377090684800237,
                7.244471327659357,
                93.09058482349712,
                0.0,
                *[50.0, 50.0, -1.0, -1.0, -1.0, -1.0],
                *[0.0],
            ],
            dtype=np.float32,
        )
        blocks = [block_1, block_2]
        expected_result = np.concatenate(blocks, dtype=np.float32)
        assert expected_result.all() == test_osp_gym._build_obs_array(test_osp_gym.controller.get_state()).all()
