import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date
import finnhub

# importing api clients
from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient, get_fmp_client
from services.alphavantage_api_client import AlphaVantage_APIClient, get_alphavantage_client

# importing plot components
from view.alphavantage_plot_components import AlphaVantage_Plot_Components
from view.finnhub_plot_components import Finnhub_Plot_Components
from view.fmp_plot_components import FMP_Plot_Components


############# PAGE CONFIG #############
st.set_page_config(
    page_title="EasyStock Learner",
    page_icon= "💹",
    layout="wide",
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

fmp_api = get_fmp_client(FMP_API_KEY)
av_api = get_alphavantage_client(AV_API_KEY)
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
    default=['MSFT', 'GOOGL'],
    key="selectbox1"
)

# The main candle chart
av_plot.draw_stock_prices(selectedTickers, av_api)

with st.popover("How to read candle stick chart? 🤔"):
    st.markdown('''
    A candlestick chart helps visualize stock price movements over time.
    Each candlestick represents a specific time period (e.g., 1 day) and shows four key prices:
    ''')
    st.markdown('''
    - Open – The price when the period started
    - Close – The price when the period ended
    - High – The highest price reached
    - Low – The lowest price reached

    Understanding Candlesticks:
    - Green Candle: The price closed higher than it opened (bullish).
    - Red Candle: The price closed lower than it opened (bearish).
    - Wicks: Thin lines above and below the candle body show the highest and lowest prices.

    Key Patterns to Watch:
    - Long green candles → Strong buying pressure
    - Long red candles → Strong selling pressure
    - Doji (small body, long wicks) → Market indecision
    - Hammer / Shooting Star → Potential trend reversal
    ''')

    st.markdown('''
    Candlestick charts help identify trends, reversals, and market sentiment quickly.
    Use them with other indicators for better analysis! 🚀
    ''')

# Smaller charts
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

st.header("Stock Rating Prediction (PLACEHOLDERS FOR NOW)")

# use the same selection as the one on top

st.subheader("AAPL")
st.write("Stock Rating: :green[Buy]")
st.write("Date: ", date.today())

st.markdown('''
    ### **Reasoning for Buy Call on Apple Stock (AAPL)**

    #### **1. Strong Revenue Growth 📈**
    - Apple’s latest earnings report shows a **15% YoY revenue increase**, driven by strong iPhone and services sales.
    - Services segment (**App Store, iCloud, Apple Music**) is growing at **20% YoY**, providing stable recurring revenue.

    #### **2. Positive Technical Indicators 📊**
    - **50-day moving average** is crossing above the **200-day moving average** (**Golden Cross**), signaling an uptrend.
    - Recent **candlestick patterns** show **higher lows**, indicating **bullish momentum**.

    #### **3. Undervalued Relative to Growth 📉**
    - Current **P/E ratio: 24** vs. historical average of **26**, suggesting slight undervaluation.
    - **Price-to-book (P/B) ratio** remains stable, reflecting confidence in asset valuation.

    #### **4. Strong Institutional Support 🏛️**
    - **Hedge funds and institutional investors** have increased holdings by **8%** in the last quarter.
    - **Warren Buffett’s Berkshire Hathaway** maintains a significant stake, showing long-term confidence.

    #### **5. Macroeconomic & Industry Trends 🌍**
    - The global shift toward **AI and AR** (**Apple Vision Pro**) positions Apple for future growth.
    - **Declining inflation** and potential **Fed rate cuts** may boost tech stock valuations.
''')
