import re
import pytest
from operation_safe_passage.policies.osp_rl import OSPReinforcementLearning


class TestOSPReinforcementLearning:

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

    TEST_PARAMS_TWO = {
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
                { "name": "uav_2", "scanners": ["default", "degraded"] },
            ],
            "num_ugvs": 2
        }
    }

    class MissionControllerSideEffect:
        def __init__(self, *functions):
            self.functions = iter(functions)

        def __call__(self, *args, **kwargs):
            function = next(self.functions)
            return function(*args, **kwargs)

    @pytest.fixture
    def test_run_gym(self, mocker):
        """
        Tests the GymAgent's init method

        Args:
            mocker (pytest.mocker): Create MagicMock objects

        Returns:
            GymAgentTest: The GymAgentTest object
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
        return OSPReinforcementLearning(max_iters=5)

    @pytest.fixture
    def test_run_gym_two(self, mocker):
        """
        Tests the GymAgent's init method with two UAVs and two UGVs

        Args:
            mocker (pytest.mocker): Create MagicMock objects

        Returns:
            OSPReinforcementLearning: The OSPReinforcementLearning object
        """
        mocked_open = mocker.mock_open(read_data="")
        mocker.patch("builtins.open", mocked_open)

        def mocked_load_network(_filename):
            return self.TEST_NETWORK

        def mocked_load_params(_filename):
            return self.TEST_PARAMS_TWO
        mocker.patch("operation_safe_passage.controller.validate_mission_params.JsonValidator.validate_config")
        mocker.patch("json.load", side_effect=self.MissionControllerSideEffect(mocked_load_network, mocked_load_params))
        mocker.patch("os.makedirs")
        return OSPReinforcementLearning(max_iters=5)

    def test_run_gym_no_ugv(self, mocker):
        """
        Tests the GymAgent's init method without a UGV present. This throws an error

        Args:
            mocker (pytest.mocker): Create MagicMock objects

        Returns:
            OSPReinforcementLearning: The OSPReinforcementLearning object
        """
        mocked_open = mocker.mock_open(read_data="")
        mocker.patch("builtins.open", mocked_open)

        def mocked_load_network(_filename):
            return self.TEST_NETWORK

        no_ugv_params = {
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
                "num_ugvs": 0
            }
        }
        def mocked_load_params(_filename):
            return no_ugv_params
        mocker.patch("operation_safe_passage.controller.validate_mission_params.JsonValidator.validate_config")
        mocker.patch("json.load", side_effect=self.MissionControllerSideEffect(mocked_load_network, mocked_load_params))
        mocker.patch("os.makedirs")
        with pytest.raises(RuntimeError, match="No UGV present; cannot navigate to goal."):
            OSPReinforcementLearning(max_iters=5)

    def test_move_idx(self, test_run_gym: OSPReinforcementLearning):
        """
        Tests the OSPReinforcementLearning's _move_idx method

        Args:
            test_mission_controller (OSPReinforcementLearning): The OSPReinforcementLearning object to test
        """
        assert 6 == test_run_gym._move_idx(None)
        assert 0 == test_run_gym._move_idx("E")

    def test_build_array_action(self, test_run_gym: OSPReinforcementLearning):
        """
        Tests the OSPReinforcementLearning's _build_array_action method

        Args:
            test_mission_controller (OSPReinforcementLearning): The OSPReinforcementLearning object to test
        """
        assert [0, 0, 1] == test_run_gym._build_array_action("E", "default", "NE")
        with pytest.raises(ValueError, match=re.escape("Scanner 'DOESNT_EXIST' not available on uav. Options: ['default', 'degraded']")):
            test_run_gym._build_array_action("E", "DOESNT_EXIST", "NE")

    def test_build_array_action_two_uav(self, test_run_gym_two: OSPReinforcementLearning):
        """
        Tests the OSPReinforcementLearning's _build_array_action method

        Args:
            test_mission_controller (OSPReinforcementLearning): The OSPReinforcementLearning object to test
        """
        assert [0, 0, 6, 2, 1, 6] == test_run_gym_two._build_array_action("E", "default", "NE")

    def test_run(self, test_run_gym: OSPReinforcementLearning, mocker):
        """
        Tests the OSPReinforcementLearning's run method

        Args:
            test_mission_controller (OSPReinforcementLearning): The OSPReinforcementLearning object to test
            monkeypatch (pytest.monkeypatch): A monkeypatch object to patch functions
        """
        mock_stdout = mocker.patch("builtins.print")
        test_run_gym.run()
        mock_stdout.assert_called_with("SUCCESS: UGV_0 reached goal (1, 1) in 2 planning steps.")
        test_run_gym.gym.max_steps = 1
        test_run_gym.run()
        mock_stdout.assert_called_with("STOP: iteration cap reached (1) before reaching goal (1, 1).")
