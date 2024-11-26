import requests
import pandas as pd
# import numpy as np
# from tomlkit.api import key

class FMP_APIClient:
    base_url = 'https://financialmodelingprep.com/api'
    # API_KEY = 'OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw'

    def get_financial_data(self, data_type, ticker, API_KEY):
        """
        Retrives financial data from FMP's API.

        This function makes an HTTP GET request to the Financial Modeling Prep (FMP) API
        to retrieve financial data for a given stock ticker. It takes three parameters:

        - data_type: The type of financial data to retrieve (e.g. "income-statement",
        "balance-sheet-statement", etc.)
        - ticker: The stock ticker symbol to get data for (e.g. "AAPL" for Apple)
        - API_KEY: Your FMP API authentication key

        The function constructs the appropriate API URL, makes the request, and returns
        the data as a pandas DataFrame. It includes error handling for API errors,
        empty responses, and unexpected issues.

        Args:
            data_type (str): type of data the user wants
            ticker (str): Ticker the user wants to retrive the data from.
            API_KEY (str): user's FMP API key

        Returns:
            pandas DataFrame: DataFrame containing requested data.
            None: If there is an error retrieving or processing the data.
        """

        url=f'{self.base_url}/v3/{data_type}/{ticker}?apikey={API_KEY}'

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

# to do: test if this works, and add all the other functions for data retrival
