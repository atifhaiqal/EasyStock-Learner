import requests
import streamlit as st
import datetime
import urllib.parse

class Alpaca_APIClient:
    """A client for interacting with the Alpaca Markets API.

    This class provides methods to fetch market data and news from Alpaca's API endpoints.

    Attributes:
        base_url (str): The base URL for the Alpaca API
        api_key_id (str): The API key ID for authentication
        api_secret_key (str): The API secret key for authentication
    """

    base_url = "https://data.alpaca.markets"

    def __init__(self, api_key_id, api_secret_key):
        """Initialize the Alpaca API client.

        Args:
            api_key_id (str): The API key ID from Alpaca
            api_secret_key (str): The API secret key from Alpaca
        """
        self.api_key_id = api_key_id
        self.api_secret_key = api_secret_key
        self.base_url = "https://data.alpaca.markets/v1beta1/news"

    def _make_request(self, params):
        """Make an HTTP GET request to the Alpaca API.

        Args:
            params (dict): Query parameters to include in the request

        Returns:
            dict: JSON response from the API

        Raises:
            RuntimeError: If there are network errors, data errors or unexpected errors
        """
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": "PKWSKNYCB90Y48I8T40S",
            "APCA-API-SECRET-KEY": "sbAZegA51WHldKlV6hLkHtgRepXDy8Yq7txtTtLm",
        }

        response = requests.get(self.base_url, params=params, headers=headers)

        if response.status_code == 200:
            return response.json().get("news", [])  # Extract "news" list from response
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    @st.cache_data(ttl=3600)
    def get_news(_self, symbols, start_date, end_date, limit=10, sort="desc"):
        """Retrieve news articles from Alpaca.

        Args:
            symbols (str or list): Stock symbol(s) to fetch news for
            start_date (str): Start date for news articles
            end_date (str): End date for news articles
            limit (int, optional): Maximum number of articles to return. Defaults to 10
            sort (str, optional): Sort order for articles ('asc' or 'desc'). Defaults to 'desc'

        Returns:
            dict: JSON response containing news articles, or None if no data
        """
        params = {
            "start": start_date,
            "end": end_date,
            "sort": sort,
            "symbols": ",".join(symbols) if isinstance(symbols, list) else symbols,
            "limit": limit,
        }

        data = _self._make_request(params)

        if data:
            return data

        return None

    def get_alpaca_datetime(self, dt):
        """Convert a datetime object to an Alpaca API compatible string format.

        Args:
            dt (datetime): Datetime object to convert

        Returns:
            str: URL-encoded datetime string in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        """
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@st.cache_resource
def get_alpaca_client(_api_key, _secret_key):
    """Create a cached instance of the Alpaca API client.

    Args:
        api_key (str): The API key from Alpaca
        secret_key (str): The secret key from Alpaca

    Returns:
        Alpaca_APIClient: An instance of the Alpaca API client
    """
    return Alpaca_APIClient(_api_key, _secret_key)
