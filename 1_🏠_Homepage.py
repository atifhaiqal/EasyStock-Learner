import streamlit as st
from config.api_settings import APIConfig
from services.fmp_api_client import FMP_APIClient
import numpy as np
import pandas as pd
import plotly.express as px

############# PAGE CONFIG #############
st.set_page_config(
    page_title="EasyStock Learner",
    page_icon= "💹",
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = "" #temporary for testing purposes

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

api_client = FMP_APIClient()
api_config = APIConfig()

# temporary value for API KEY
st.session_state["api_key"] = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"

############# PAGE STARTS HERE #############

st.title(":green[EasyStock] Learner :chart:")

if st.session_state["user_name"] and "" or st.session_state["api_key"] != "":
    st.write("## Hi ", st.session_state["user_name"], "!")
else:
    st.write("## Please proceed to the profile page to set up API key and nickname")

st.markdown("## Pick stocks to compare")

ticker = st.selectbox(
    "Select ticker:",
    api_config.get_ticker_options(),
)

financial_data_type = st.selectbox(
    "Financial data type",
    api_config.get_financial_data_options(),
    index=None,
    placeholder="Select financial data type"
)

st.markdown(f"You selected: :green[{financial_data_type}]")

if financial_data_type == "Historical Price smaller interval":
    interval = st.selectbox(
        "Interval",
        api_config.get_interval_options(),
        index=1
    )
    financial_data_type = f'historical-chart/{interval}'

# base_url = 'https://financialmodelingprep.com/api'
# ticker = "AAPL"
# data_type = "income-statement"

df = api_client.get_financial_data(financial_data_type, ticker, st.session_state["api_key"])

# to do
# make a view class
# api_interface <---> streamline_page.py <---> view_class

st.dataframe(df)
