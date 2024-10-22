import os
import json
from .logging_config import setup_logging 
from ..model_objects.config_model import ConfigModel

logger = setup_logging()

class ConfigHelper:
    def __init__(self):
        self.settings = None 

        possibility_config_locations = [
            os.path.join('api_key.json'),
        ]

        file = self._get_config(possibility_config_locations)
        if file is not None:
            self.settings = file.model_dump()

        if not self.settings:
            raise ValueError("Configuration file variable not found!")
        
    def _get_config(self, list_path: list):
        for config_location in list_path:
            if os.path.exists(config_location): 
                with open(config_location, 'r') as file: 
                    return ConfigModel.model_validate(json.load(file))
        return None

if __name__ == "__main__":
    config = ConfigHelper()
    print(config.settings)
