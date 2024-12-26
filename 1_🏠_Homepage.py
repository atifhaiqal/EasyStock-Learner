import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import finnhub

from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient
from services.alphavantage_api_client import AlphaVantage_APIClient

from view.alphavantage_plot_components import AlphaVantage_Plot_Components
from view.finnhub_plot_components import Finnhub_Plot_Components
from view.fmp_plot_components import FMP_Plot_Components


############# PAGE CONFIG #############
st.set_page_config(
    page_title="EasyStock Learner",
    page_icon= "💹",
    layout="wide"
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = "" #temporary for testing purposes

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

#temp
FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'
FINNHUB_API_KEY = 'ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg'

fmp_api = FMP_APIClient(FMP_API_KEY)
av_api = AlphaVantage_APIClient(AV_API_KEY)
finnhub_client = finnhub.Client(FINNHUB_API_KEY)
api_config = FMP_APIConfig()

fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()
av_plot = AlphaVantage_Plot_Components()
# temporary value for API KEY
st.session_state["api_key"] = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"

############# PAGE STARTS HERE #############

st.title(":green[EasyStock] Learner :chart:")

if st.session_state["user_name"] and "" or st.session_state["api_key"] != "":
    st.write("## Hi ", st.session_state["user_name"], "!")
else:
    st.write("## Please proceed to the profile page to set up API key and nickname")

st.markdown("## Pick stocks to compare")

selectedTickers = st.multiselect(
    "Select ticker:",
    api_config.get_ticker_options(),
    default=['AAPL', 'GOOGL'],
    key="selectbox1"
)

av_plot.draw_stock_prices(selectedTickers, av_api)

col1, col2, col3 = st.columns(3)

with col1:
    fmp_plot.draw_revenue(selectedTickers, fmp_api)

    fin_plot.draw_pe_ratio(selectedTickers, finnhub_client)

with col2:
    fmp_plot.draw_ebitda(selectedTickers, fmp_api)

    fin_plot.draw_pb_ratio(selectedTickers, finnhub_client)
with col3:
    fin_plot.draw_dividend_yield_annual(selectedTickers, finnhub_client)

    fin_plot.draw_eps_ratio(selectedTickers, finnhub_client)
