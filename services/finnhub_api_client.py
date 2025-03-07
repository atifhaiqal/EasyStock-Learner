import pandas as pd
import streamlit as st
import finnhub

class Finnhub_APIClient:

    def __init__(self, api_key):
        self.api_key = api_key
        self.finnhub_client = finnhub.Client(api_key)

    @st.cache_resource
    def get_company_basic_financials(_self, symbol, metric):
        return _self.finnhub_client.company_basic_financials(symbol, metric)

    @st.cache_resource
    def get_recommendation_trends(_self, ticker):
        return _self.finnhub_client.recommendation_trends(ticker)

    def get_rating_consensus(_self, ticker):
        rating = _self.get_recommendation_trends(ticker)
        ratings = rating[0]

        weights = {'strongBuy': 5, 'buy': 4, 'hold': 3, 'sell': 2, 'strongSell': 1}

        total_weighted_score = 0
        total_ratings = 0
        for key, weight in weights.items():
            total_weighted_score += ratings[key] * weight
            total_ratings += ratings[key]

        if total_ratings == 0:
            return None  # Avoid division by zero

        consensus_score = total_weighted_score / total_ratings

        # Classify consensus
        if consensus_score >= 4.5:
            consensus_label = "Strong Buy"
        elif consensus_score >= 3.5:
            consensus_label = "Buy"
        elif consensus_score >= 2.5:
            consensus_label = "Hold"
        elif consensus_score >= 1.5:
            consensus_label = "Sell"
        else:
            consensus_label = "Strong Sell"

        return consensus_label


@st.cache_resource
def get_finnhub_client(api_key):
    """Streamlit cached resource for the Finnhub API client."""
    return Finnhub_APIClient(api_key)
