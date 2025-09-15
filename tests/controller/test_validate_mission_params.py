import jsonschema
import pytest
from operation_safe_passage.controller.validate_mission_params import JsonValidator


class TestJsonValidator:

    TEST_PARAMS = {

        "processing_params": {
            "UGV_traversal_time": 20,
            "UGV_clear_time": 60,
            "UAV_traversal_time": 1
        },
        "environment": {
            "num_nodes" : 4,
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
        },
        "terrain_coeff": {"Grassy": 1.0, "Sandy": 0.0},
        "scanner_params": {
            "scanners": {
                "default": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":1},
                "degraded": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":1},
                "baseline": {"visibility_scale": 0.55, "visibility_metric": 6, "kappa": 3.0, "noise_scale": 3.0, "noise_std": 0.75, "threshold": 0.5, "time":30}
            }
        },
        "agents": {
            "uavs": [
                {
                    "name": "uav",
                    "scanners": ["default", "degraded"]
                },
            ],
            "num_ugvs": 1
        }
    }

    def test_json_validator(self, mocker):
        """
        Tests the JsonValidator

        Args:
            mocker (pytest.mocker): Used to create Mock objects
        """
        mock_stdout = mocker.patch("builtins.print")
        mocked_open = mocker.mock_open(read_data="")
        mocker.patch("builtins.open", mocked_open)
        mocker.patch("json.load", return_value=self.TEST_PARAMS)
        validator = JsonValidator()
        validator.validate_config("test/path", silent=False) # Test valid JSON
        mock_stdout.assert_called_with("JSON data is valid.")

        mocker.patch("json.load", return_value={"processing_params": "Wrong type"}) # Test invalid JSON
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator.validate_config("test/path", silent=False)
            mock_stdout.assert_called_with("JSON data is invalid: ")
