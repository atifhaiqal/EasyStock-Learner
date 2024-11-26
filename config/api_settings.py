import json
import os

class APIConfig:
    def __init__(self, config_file='config/api_config.json'):
        # Check if the config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"The configuration file {config_file} does not exist!")

        # Load the JSON configuration file
        with open(config_file, 'r') as f:
            self.config_data = json.load(f)

    def get_ticker_options(self):
        """Returns the list of ticker options from the configuration."""
        return self.config_data.get("ticker_options", [])

    def get_financial_data_options(self):
        """Returns the list of financial data options from the configuration."""
        return self.config_data.get("financial_data_options", [])

    def get_interval_options(self):
        """Returns the list of interval options from the configuration."""
        return self.config_data.get("interval_options", [])

    def display_config(self):
        """Optional: Display the entire configuration."""
        print(json.dumps(self.config_data, indent=4))
