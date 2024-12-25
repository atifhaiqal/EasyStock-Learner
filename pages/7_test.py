# THIS IS PURELY FOR TEST PURPOSES, DO NOT USE IN PRODUCTION

from streamlit.elements import metric
import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from view.finnhub_plot_components import Finnhub_Plot_Components
from view.fmp_plot_components import FMP_Plot_Components
from services.fmp_api_client import FMP_APIClient
from config.api_settings import YF_APIConfig
import finnhub

# Page config
st.set_page_config(
    page_title="TEST",
)

FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"

api_config = YF_APIConfig()
fmp_api = FMP_APIClient(FMP_API_KEY)
fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()


############# PAGE STARTS HERE #############

st.title("TESTING PAGE")

finnhub_client = finnhub.Client(api_key="ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg")

selectedTickers = st.multiselect(
    "Select ticker:",
    api_config.get_ticker_options(),
    key="selectbox1"
)

# metrics = finnhub_client.company_basic_financials('AAPL', 'all')
# st.write(metrics)

# metrics = finnhub_client.company_basic_financials('AAPL', 'all')
# dividendYield = metrics["metric"]["dividendYieldIndicatedAnnual"]
# st.write(dividendYield)

# fin_plot.draw_dividend_yield_annual(selectedTickers, finnhub_client)

# incomeStatement = fmp_api.get_income_statement("AAPL")

# st.write(incomeStatement)

fmp_plot.draw_revenue(selectedTickers, fmp_api)
