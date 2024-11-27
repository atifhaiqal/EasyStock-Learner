import requests
import pandas as pd
# import numpy as np
# from tomlkit.api import key

class AV_APIClient:
    base_url = 'https://www.alphavantage.co/query?'
    # API_KEY = 'WGHKWKAR5TGFV4IC'

    def get_financial_data(self, data_type, ticker, API_KEY):
        """
        Retrieves financial data from Alpha Vantage API.

        This function makes an HTTP GET request to the Alpha Vantage API
        to retrieve financial data for a given stock ticker. It takes three parameters:

        - data_type: The type of financial data to retrieve (e.g. "TIME_SERIES_DAILY",
        "GLOBAL_QUOTE", etc.)
        - ticker: The stock ticker symbol to get data for (e.g. "AAPL" for Apple)
        - API_KEY: Your Alpha Vantage API authentication key

        The function constructs the appropriate API URL, makes the request, and returns
        the data as a pandas DataFrame. It includes error handling for API errors,
        empty responses, and unexpected issues.

        Args:
            data_type (str): Alpha Vantage API function name for the data type requested
            ticker (str): Ticker symbol to retrieve the data for
            API_KEY (str): User's Alpha Vantage API key

        Returns:
            pandas DataFrame: DataFrame containing requested data.
            None: If there is an error retrieving or processing the data.
        """

        url=f'{self.base_url}function={data_type}&symbol={ticker}&apikey={API_KEY}'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()

            if not data:  # Check if data is empty
                raise ValueError(f"No data returned for {ticker}")

            df = pd.DataFrame(data)
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return None
        except ValueError as e:
            print(f"Error processing data: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def get_company_overview(self, ticker, API_KEY):
        """
        Retrieves company overview data from Alpha Vantage API for a given ticker.

        This function makes an HTTP GET request to the Alpha Vantage API
        to retrieve overview information like market cap, PE ratio, dividend yield etc.
        for a given stock ticker. It takes two parameters and returns the data as
        a pandas DataFrame.

        Args:
            ticker (str): The stock ticker symbol to get overview data for (e.g. "AAPL")
            API_KEY (str): Alpha Vantage API authentication key

        Returns:
            pandas DataFrame: DataFrame containing company overview data.
            None: If there is an error retrieving or processing the data.
        """

        url=f'{self.base_url}function=OVERVIEW&symbol={ticker}&apikey={API_KEY}'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()

            if not data:  # Check if data is empty
                raise ValueError(f"No data returned for {ticker}")

            df = pd.DataFrame(data)
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return None
        except ValueError as e:
            print(f"Error processing data: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
