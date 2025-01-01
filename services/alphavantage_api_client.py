import requests
import pandas as pd
import streamlit as st

# API_KEY = 'WGHKWKAR5TGFV4IC'
class AlphaVantage_APIClient:
    """Alpha Vantage API Client with centralized request logic and caching."""

    base_url = "https://www.alphavantage.co/query?"

    def __init__(self, api_key):
        self.api_key = api_key

    def _make_request(self, params):
        """Centralized API request logic."""
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data or "Error Message" in data:
                raise ValueError(f"API returned an error: {data.get('Error Message', 'Unknown error')}")

            return data
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error: {e}")
        except ValueError as e:
            raise RuntimeError(f"Data error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

    @st.cache_data(ttl=3600)
    def get_financial_data(_self, data_type, ticker):
        """Retrieve and cache financial data."""
        params = {
            "function": data_type,
            "symbol": ticker,
            "apikey": _self.api_key,
        }
        data = _self._make_request(params)

        if data is not None:
            return pd.DataFrame(data)

        return None

    @st.cache_data(ttl=3600)
    def get_company_overview(_self, ticker):
        """Retrieve and cache company overview."""
        params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": _self.api_key,
        }
        data = _self._make_request(params)

        if data is not None:
            return pd.DataFrame([data])  # Convert dict to DataFrame for consistency

        return None

    @st.cache_data(ttl=3600)
    def get_time_series_stock_prices(_self, ticker):
        """Retrieve and cache daily time series stock prices."""

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": _self.api_key,
        }
        data = _self._make_request(params)

        if data is not None and "Time Series (Daily)" in data:
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
            return df.sort_index()

        return None

# Cache the AlphaVantage_APIClient instance based on API key
@st.cache_resource
def get_alphavantage_client(api_key):
    """Cache the AlphaVantageAPIClient instance."""
    return AlphaVantage_APIClient(api_key)
