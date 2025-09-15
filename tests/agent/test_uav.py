import pytest
from dataclasses import dataclass
from operation_safe_passage.agent.uav import UAV
from unittest.mock import patch


class TestUAV:

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
        }
    }

    @pytest.fixture
    def test_uav(self, mocker):
        """
        Tests the UAV's __init__ method

        Args:
            mocker (pytest.Mocker): Tool to create Mock objects

        Returns:
            UAV: The created UAV object
        """
        empty_uav = UAV()
        assert {"default": dict(empty_uav.acc_defaults)} == empty_uav.scanners

        uav = UAV(
            scanners_map=self.TEST_PARAMS["scanner_params"]["scanners"],
            terrain_coeff=self.TEST_PARAMS["terrain_coeff"],
            traversal_time=self.TEST_PARAMS["processing_params"]["UAV_traversal_time"]
        )
        scanner_dict = {
            "default": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":1},
            "degraded": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":1},
            "baseline": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":30}
        }
        assert scanner_dict == uav.scanners
        return uav

    def test_list_scanners(self, test_uav: UAV):
        """
        Tests the UAV's list_scanners method

        Args:
            test_uav (UAV): The UAV object retured from the fixture
        """
        assert ["default", "degraded", "baseline"] == test_uav.list_scanners()

    @dataclass
    class ResolveScannerNameTestCase:
        """
        A dataclass to help test the _resolve_scanner_name test cases
        """
        scanner_name: str
        expected_result: str
        id: str
        expected_error: type[Exception] = None
        error_message: str | None = None

    @pytest.mark.parametrize(
        "test_case",
        [
            ResolveScannerNameTestCase(None, "default", "test_resolve_scanner_name_method_none"),
            ResolveScannerNameTestCase("default", "default", "test_resolve_scanner_name_default_scanner"),
            ResolveScannerNameTestCase(0, "default", "test_resolve_scanner_name_default_with_index"),
            ResolveScannerNameTestCase("degraded", "degraded", "test_resolve_scanner_name_degraded"),
            ResolveScannerNameTestCase(1, "degraded", "test_resolve_scanner_name_degraded_with_index"),
            ResolveScannerNameTestCase(99, None, "test_resolve_scanner_name_out_of_bounds", IndexError, "Scanner index out of range: 99"),
            ResolveScannerNameTestCase("doesnt_exist", None, "test_resolve_scanner_name_scanner_doesnt_exist", KeyError, "Unknown scanner: doesnt_exist"),
            ResolveScannerNameTestCase([], None, "test_resolve_scanner_name_wrong_type", TypeError, "scan method must be None, str, or int."),
            ResolveScannerNameTestCase("REMOVE_SCANNERS", None, "test_resolve_scanner_name_no_scanners", RuntimeError, "No scanners available.")
        ],
        ids=lambda tc: tc.id
    )
    def test_resolve_scanner_name(self, test_uav: UAV, test_case: ResolveScannerNameTestCase):
        """
        Tests the UAV's _resolve_scanner_name method

        Args:
            test_uav (UAV): The UAV object retured from the fixture
            test_case (ResolveScannerNameTestCase): The current test case to run
        """
        if test_case.scanner_name == "REMOVE_SCANNERS":
            test_uav.scanners = {}
        if test_case.expected_error:
            with pytest.raises(test_case.expected_error, match=test_case.error_message):
                test_uav._resolve_scanner_name(test_case.scanner_name)
        else:
            assert test_uav._resolve_scanner_name(test_case.scanner_name) == test_case.expected_result

    @dataclass
    class ScanTestCase:
        """
        A dataclass to help test the scan test cases
        """
        node: dict
        expected_result: float
        id: str

    @pytest.mark.parametrize(
        "test_case",
        [
            ScanTestCase(None, 0.0, "test_scan_node_is_node"),
            ScanTestCase({"id": "(1, 1)"}, 0.0, "test_scan_wrong_id"),
            ScanTestCase({"id": "(0, 0)", "REMOVE_SCANNERS": True}, 0.0, "test_scan_no_scanners"),
            ScanTestCase(
                {
                    "id": "(0, 0)",
                    "metadata": {
                        "temperature": 60,
                        "wind_speed": 50,
                        "visibility": 50,
                        "precipitation": 10,
                        "terrain_type": "Grassy",
                    },
                    "inaccessible": {}
                },
                (0.75, 0.80),
                "test_normal_scan"
            )
        ],
        ids=lambda tc: tc.id
    )
    @patch.object(UAV, "_scan_one", return_value=(0.75, 0.80))
    def test_scan(self, _mock_scan_one, test_uav: UAV, test_case):
        """
        Tests the UAV's scan method

        Args:
            _mock_scan_one (MagicMock): Mock object for _scan_one method
            test_uav (UAV): The UAV object retured from the fixture
            test_case (ScanTestCase): The test case to run
        """
        if test_case.node and "REMOVE_SCANNERS" in test_case.node:
            test_uav.scanners = []
        test_uav.current_node = "(0, 0)"
        if test_case.expected_result == 0.0:
            assert test_case.expected_result == test_uav.scan(test_case.node, "default")
        else:
            estimate, node = test_uav.scan(test_case.node, "default")
            assert test_case.expected_result[0] == estimate
            assert test_case.expected_result[1] == node['inaccessible']["uav_accuracy"]

    @pytest.mark.parametrize(
        "test_case",
        [
            ScanTestCase({"metadata": None}, (0.0, 0.0), "test_scan_none_meta"),
            ScanTestCase(
                {
                    "metadata": {
                        "terrain": "Grassy",
                        "temperature": 60,
                        "wind_speed": 50,
                        "visibility": 50,
                        "precipitation": 10,
                        "terrain_type": "Grassy"
                    },
                    "inaccessible": {"mine_presence": False}
                },
                (0.59, 0.61),
                "test_scan_one"
            )
        ]
    )
    @patch.object(UAV, "_compute_accuracy", return_value=0.75)
    @patch.object(UAV, "_compute_estimate", return_value=0.60)
    def test_scan_one(self, _mock_compute_accuracy, _mock_compute_estimate, test_uav: UAV, test_case):
        """
        Tests the UAV's scan_one method

        Args:
            _mock_scan_one (MagicMock): Mock object for _compute_accuracy method
            _mock_scan_one (MagicMock): Mock object for _compute_estimate method
            test_uav (UAV): The UAV object retured from the fixture
            test_case (ScanTestCase): The test case to run
        """
        if test_case.expected_result[0] == test_case.expected_result[1] == 0:
            assert 0.0 == test_uav._scan_one(test_case.node, test_uav.acc_defaults)
        else:
            average_value = 0
            num_epochs = 1000
            for _i in range(0, num_epochs):
                estimate, _accuracy = test_uav._scan_one(test_case.node, test_uav.acc_defaults)
                average_value += estimate
            average_value /= num_epochs
            assert test_case.expected_result[0] <= average_value <= test_case.expected_result[1]

    def test_scan_idx_for_uav(self, test_uav: UAV):
        """
        Tests the UAV's scan_idx_for_uav method

        Args:
            test_uav (UAV): The UAV object retured from the fixture
        """
        assert 0 == test_uav.scan_idx_for_uav("default")
        assert 0 == test_uav.scan_idx_for_uav("doesn't_exist")
        assert 0 == test_uav.scan_idx_for_uav(None)
        assert 1 == test_uav.scan_idx_for_uav("degraded")
        test_uav.scanners = {}
        assert 0 == test_uav.scan_idx_for_uav("degraded")

    def test_compute_accuracy(self, test_uav: UAV):
        """
        Tests the UAV's compute_accuracy method

        Args:
            test_uav (UAV): The UAV object retured from the fixture
        """
        average_value = 0
        num_epochs = 1000
        for _i in range(0, num_epochs):
            average_value += test_uav._compute_accuracy(
                                temperature=60,
                                wind_speed=50,
                                visibility=50,
                                precipitation=10,
                                terrain_type="Grassy",
                                params=test_uav.acc_defaults
                            )
        average_value /= num_epochs
        assert 0.39 <= average_value <= 0.41

    def test_compute_estimate(self, test_uav: UAV):
        """
        Tests the UAV's compute_estimate method

        Args:
            test_uav (UAV): The UAV object retured from the fixture
        """
        average_value = 0
        num_epochs = 1000
        for _i in range(0, num_epochs):
            average_value += test_uav._compute_estimate(
                                accuracy=0.75,
                                ground_truth=True,
                                params=test_uav.acc_defaults
                            )
        average_value /= num_epochs
        assert 0.66 <= average_value <= 0.71
