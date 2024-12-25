import requests
import pandas as pd

class FMP_APIClient:
    _instance = None
    base_url = 'https://financialmodelingprep.com/api'

    def __new__(cls, api_key):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.api_key = api_key
        return cls._instance

    def __init__(self, api_key):
        self.api_key = api_key

    def get_financial_data(self, data_type, ticker):
        """
        Retrives financial data from FMP's API.

        This function makes an HTTP GET request to the Financial Modeling Prep (FMP) API
        to retrieve financial data for a given stock ticker. It takes two parameters:

        - data_type: The type of financial data to retrieve (e.g. "income-statement",
        "balance-sheet-statement", etc.)
        - ticker: The stock ticker symbol to get data for (e.g. "AAPL" for Apple)

        The function constructs the appropriate API URL, makes the request, and returns
        the data as a pandas DataFrame. It includes error handling for API errors,
        empty responses, and unexpected issues.

        Args:
            data_type (str): type of data the user wants
            ticker (str): Ticker the user wants to retrive the data from.

        Returns:
            pandas DataFrame: DataFrame containing requested data.
            None: If there is an error retrieving or processing the data.
        """

        url=f'{self.base_url}/v3/{data_type}/{ticker}?apikey={self.api_key}'

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

    def get_financial_ratio(self, ticker):
        """
        Retrieves financial ratios from FMP's API for a given ticker.

        This function makes an HTTP GET request to the Financial Modeling Prep (FMP) API
        to retrieve trailing twelve month financial ratios for a given stock ticker.
        It takes one parameter and returns the data as a pandas DataFrame.

        Args:
            ticker (str): The stock ticker symbol to get ratios for (e.g. "AAPL")

        Returns:
            pandas DataFrame: DataFrame containing financial ratios.
            None: If there is an error retrieving or processing the data.
        """

        url=f'{self.base_url}/v3/ratios-ttm/{ticker}?apikey={self.api_key}'

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

    def get_key_metrics(self, ticker):
        """
        Retrieves key metrics from FMP's API for a given ticker.

        This function makes an HTTP GET request to the Financial Modeling Prep (FMP) API
        to retrieve annual key metrics for a given stock ticker. It takes one parameter
        and returns the data as a pandas DataFrame.

        Args:
            ticker (str): The stock ticker symbol to get key metrics for (e.g. "AAPL")

        Returns:
            pandas DataFrame: DataFrame containing key metrics data.
            None: If there is an error retrieving or processing the data.
        """

        url=f'{self.base_url}/v3/key-metrics/{ticker}?period=annual&apikey={self.api_key}'

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

    def get_income_statement(self, ticker):
        """
        Retrieves annual income statement data from FMP's API for a given ticker.

        This function makes an HTTP GET request to the Financial Modeling Prep (FMP) API
        to retrieve annual income statement data for a given stock ticker. It takes one
        parameter and returns the data as a pandas DataFrame.

        Args:
            ticker (str): The stock ticker symbol to get income statement for (e.g. "AAPL")

        Returns:
            pandas DataFrame: DataFrame containing income statement data.
            None: If there is an error retrieving or processing the data.
        """

        url=f'{self.base_url}/v3/income-statement/{ticker}?period=annual&apikey={self.api_key}'

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
