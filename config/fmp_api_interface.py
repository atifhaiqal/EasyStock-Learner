import requests
import pandas as pd
import numpy as np
from tomlkit.api import key

base_url = 'https://financialmodelingprep.com/api'
API_KEY = 'OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw'

def get_financial_data(data_type, ticker, API_KEY):
    """
    Retrives financial data from FMP's API.

    Args:
        data_type (str): type of data the user wants
        ticker (str): Ticker the user wants to retrive the data from.
        API_KEY (str): user's FMP API key

    Returns:
        pandas DataFrame: DataFrame containing requested data.
    """

    base_url = 'https://financialmodelingprep.com/api'
    url=f'{base_url}/v3/{data_type}/{ticker}?apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data)
    return df

# to do: test if this works, and add all the other functions for data retrival
