import requests
import pandas as pd
import streamlit as st

class FMP_APIClient:
    """FMP API Client with centralized request logic and caching."""

    base_url = "https://financialmodelingprep.com/api"

    def __init__(self, api_key):
        self.api_key = api_key

    def _make_request(self, endpoint, params=None):
        """
        Centralized API request logic.
        """
        url = f"{self.base_url}/{endpoint}"

        if not params:
            params = {}
        params["apikey"] = self.api_key

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                raise ValueError("No data returned from the API.")

            return data
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error: {e}")
        except ValueError as e:
            raise RuntimeError(f"Data error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")


    @st.cache_data(ttl=3600)  # Cache API responses for 1 hour
    def get_financial_data(_self, data_type, ticker):
        """Retrieves financial data from FMP's API."""
        endpoint = f"v3/{data_type}/{ticker}"
        data = _self._make_request(endpoint)

        if data is not None:
            return pd.DataFrame(data)

        return None

    @st.cache_data(ttl=3600)
    def get_financial_ratios(_self, ticker):
        """Retrieves financial ratios data."""
        endpoint = f"v3/ratios-ttm/{ticker}"
        data = _self._make_request(endpoint)

        if data is not None:
            return pd.DataFrame(data)

        return None

    @st.cache_data(ttl=3600)
    def get_key_metrics(_self, ticker):
        """Retrieves key metrics data."""
        endpoint = f"v3/key-metrics/{ticker}"
        params = {"period": "annual"}
        data = _self._make_request(endpoint, params)

        if data is not None:
            return pd.DataFrame(data)

        return None

    @st.cache_data(ttl=3600)
    def get_income_statement(_self, ticker):
        """Retrieves income statement data."""
        endpoint = f"v3/income-statement/{ticker}"
        params = {"period": "annual"}
        data = _self._make_request(endpoint, params)

        if data is not None:
            return pd.DataFrame(data)

        return None

    def get_balance_sheet(_self, ticker):
        endpoint = f"v3/balance-sheet-statement/{ticker}"
        params = {"period": "annual"}
        data = _self._make_request(endpoint, params)

        if data is not None:
            return pd.DataFrame(data)

        return None

@st.cache_resource
def get_fmp_client(api_key):
    """Streamlit cached resource for the FMP API client."""
    return FMP_APIClient(api_key)
