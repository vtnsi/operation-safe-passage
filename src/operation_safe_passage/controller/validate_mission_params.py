import json
import jsonschema


class JsonValidator:
    def __init__(self):
        """
        The constructor which sets the schema for the JSON
        """
        self.schema = {
            "type": "object",
            "properties": {
                "processing_params": {
                    "type": "object",
                    "properties": {
                        "UGV_traversal_time": {
                            "type": "integer",
                        },
                        "UGV_clear_time": {
                            "type": "integer",
                        },
                        "UAV_traversal_time": {
                            "type": "integer",
                        }
                    },
                    "required": [
                        "UGV_traversal_time",
                        "UGV_clear_time",
                        "UAV_traversal_time"
                    ]     
                },
                "environment": {
                    "type": "object",
                    "properties": {
                        "num_nodes": {
                            "type": "integer"
                        },
                        "mine_likelihood": {
                            "type": "number"
                        },
                        "terrain_types": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "transition_matrix": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            }
                        },
                        "precipitation_params": {
                            "type": "object",
                            "patternProperties": {
                                "^[A-Za-z0-9 -_]+": {
                                    "type": "object",
                                    "properties": {
                                        "rain_probability": {
                                            "type": "number"
                                        },
                                        "scale": {
                                            "type": "integer"
                                        }
                                    },
                                    "required": [
                                        "rain_probability",
                                        "scale"
                                    ]
                                }
                            }
                        },
                        "temp_params": {
                            "type": "object",
                            "patternProperties": {
                                "^[A-Za-z0-9 -_]+": {
                                    "type": "object",
                                    "properties": {
                                        "base_temp": {
                                            "type": "integer"
                                        },
                                        "diurnal_variation": {
                                            "type": "integer"
                                        },
                                        "precipitation_effect": {
                                            "type": "number"
                                        },
                                    },
                                    "required": [
                                        "base_temp",
                                        "diurnal_variation",
                                        "precipitation_effect"
                                    ]
                                }
                            }
                        },
                        "wind_params": {
                            "type": "object",
                            "patternProperties": {
                                "^[A-Za-z0-9 -_]+": {
                                    "type": "object",
                                    "properties": {
                                        "base_wind": {
                                            "type": "integer"
                                        },
                                        "time_variation": {
                                            "type": "integer"
                                        },
                                        "precipitation_effect": {
                                            "type": "number"
                                        }
                                    },
                                    "required": [
                                        "base_wind",
                                        "time_variation",
                                        "precipitation_effect"
                                    ]
                                }
                            }
                        },
                        "visibility_params": {
                            "type": "object",
                            "patternProperties": {
                                "^[A-Za-z0-9 -_]+": {
                                    "type": "object",
                                    "properties": {
                                        "max_visibility": {
                                            "type": "integer"
                                        },
                                        "time_effect": {
                                            "type": "integer"
                                        },
                                        "precipitation_effect": {
                                            "type": "number"
                                        },
                                    },
                                    "required": [
                                        "max_visibility",
                                        "time_effect",
                                        "precipitation_effect"
                                    ]
                                }
                            }
                        }
                    },
                    "required": [
                        "num_nodes",
                        "mine_likelihood",
                        "terrain_types",
                        "transition_matrix",
                        "precipitation_params",
                        "temp_params",
                        "wind_params",
                        "visibility_params"
                    ] 
                },
                "terrain_coeff": {
                    "type": "object"
                },
                "scanner_params": {
                    "type": "object",
                    "properties": {
                        "scanners": {
                            "type": "object",
                            "patternProperties": {
                                "^[A-Za-z0-9 -_]+": {
                                    "type": "object",
                                    "properties": {
                                        "visibility_scale": {
                                            "type": "number"
                                        },
                                        "visibility_metric": {
                                            "type": "integer"
                                        },
                                        "kappa": {
                                            "type": "number"
                                        },
                                        "noise_scale": {
                                            "type": "number"
                                        },
                                        "noise_std": {
                                            "type": "number"
                                        },
                                        "threshold": {
                                            "type": "number"
                                        },
                                        "time": {
                                            "type": "integer"
                                        }
                                    },
                                    "required": [
                                        "visibility_scale",
                                        "visibility_metric",
                                        "kappa",
                                        "noise_scale",
                                        "noise_std",
                                        "threshold",
                                        "time"
                                    ] 
                                }
                            }
                        }
                    },
                    "required": [
                        "scanners"
                    ]
                },
                "agents": {
                    "type": "object",
                    "properties": {
                        "uavs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string"
                                    },
                                    "scanners": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                        }
                                    }
                                },
                                "required": [
                                    "name",
                                    "scanners"
                                ]
                            }
                        },
                        "num_ugvs": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "uavs",
                        "num_ugvs"
                    ] 
                },
                "additionalProperties": False
            }
        }

    def validate_config(self, config_file: str, silent: bool = True):
        """
        Validates a JSON config file containing killweb information
        
        Parameters:
            config_file (str): The config file to validate
            silent (bool): True if MIMIK is running in silent mode
        
        Returns:
            True if the config file had valid JSON schema
        """
        with open(config_file, 'r') as file:
            data = json.load(file)
            try:
                jsonschema.validate(instance=data, schema=self.schema)
                if not silent:
                    print("JSON data is valid.")
            except jsonschema.exceptions.ValidationError as e:
                print(f"JSON data is invalid: {e.message}")
                raise e
