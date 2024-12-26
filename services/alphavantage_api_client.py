import requests
import pandas as pd
# import numpy as np
# from tomlkit.api import key

class AlphaVantage_APIClient:
    # API_KEY = 'WGHKWKAR5TGFV4IC'
    _instance = None
    base_url = 'https://www.alphavantage.co/query?'

    def __new__(cls, api_key):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.api_key = api_key
        return cls._instance

    def __init__(self, api_key):
        self.api_key = api_key

    def get_financial_data(self, data_type, ticker):
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

        url=f'{self.base_url}function={data_type}&symbol={ticker}&apikey={self.api_key}'

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

    def get_company_overview(self, ticker):
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

        url=f'{self.base_url}function=OVERVIEW&symbol={ticker}&apikey={self.api_key}'

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

    def get_time_series_stock_prices(self, ticker):
        """
        Retrieves daily time series stock price data from Alpha Vantage API.

        This function makes an HTTP GET request to the Alpha Vantage API
        to retrieve daily stock price data (open, high, low, close, volume)
        for a given stock ticker. The data is returned in compact form with
        the most recent 100 data points.

        Args:
            ticker (str): The stock ticker symbol to get price data for (e.g. "AAPL")

        Returns:
            pandas DataFrame: DataFrame containing daily stock price data.
            None: If there is an error retrieving or processing the data.
        """
        # Initialize cache if it doesn't exist
        if not hasattr(self, '_cache'):
            self._cache = {}

        # Get current date
        today = pd.Timestamp.now().date()

        # Check if data is in cache and from today
        if ticker in self._cache and self._cache[ticker]['date'] == today:
            return self._cache[ticker]['data']

        url=f'{self.base_url}function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={self.api_key}'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()

            # Check if the 'Time Series (Daily)' key exists in the response
            if "Time Series (Daily)" not in data:
                raise ValueError(f"No time series data found for {ticker}")

            # Extract the time series data
            time_series = data["Time Series (Daily)"]

            # Convert time series to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)  # Convert the index to datetime
            df.columns = ["Open", "High", "Low", "Close", "Volume"]  # Rename columns for readability

            # Sort the data by date (ascending order)
            df = df.sort_index()

            # Cache the data
            self._cache[ticker] = {
                'date': today,
                'data': df
            }

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
