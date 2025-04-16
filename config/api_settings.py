import json
import os

class FMP_APIConfig:
    def __init__(self, config_file='config/fmp_api_config.json'):
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

class YF_APIConfig:
    def __init__(self, config_file='config/yf_api_config.json'):
        # Check if the config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"The configuration file {config_file} does not exist!")

        # Load the JSON configuration file
        with open(config_file, 'r') as f:
            self.config_data = json.load(f)

    def get_ticker_options(self):
        """Returns the list of ticker options from the configuration."""
        return self.config_data.get("ticker_options", [])

class APIConfig:
    def __init__(self, config_file='config/api_key.json'):
        # Check if the config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"The configuration file {config_file} does not exist!")

        # Load the JSON configuration file
        with open(config_file, 'r') as f:
            self.config_data = json.load(f)

    def get_alpaca_api_key(self):
        """Returns the API Key for Alpaca"""
        return self.config_data.get("alpaca_api_key", [])

    def get_alpaca_secret_key(self):
        """Returns the Secret Key for Alpaca LM"""
        return self.config_data.get("alpaca_secret_key", [])
    
    def get_fmp_api_key(self):
        """Returns the API Key for Financial Modeling Prep (FMP)"""
        return self.config_data.get("fmp_api_key")

    def get_av_api_key(self):
        """Returns the API Key for Alpha Vantage (AV)"""
        return self.config_data.get("av_api_key")

    def get_finnhub_api_key(self):
        """Returns the API Key for Finnhub"""
        return self.config_data.get("finnhub_api_key")

    def get_gemini_api_key(self):
        """Returns the API Key for Gemini"""
        return self.config_data.get("gemini_api_key")
