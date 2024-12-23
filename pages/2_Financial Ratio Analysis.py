import streamlit as st
from config.api_settings import FMP_APIConfig
from services.av_api_client import AV_APIClient
from services.fmp_api_client import FMP_APIClient
from view.plot_components import Plot_Components
# import numpy as np
# import pandas as pd
# import plotly.express as px

############# PAGE CONFIG #############
st.set_page_config(
    page_title="Financial Ratio Analysis",
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

api_client = FMP_APIClient()
api_config = FMP_APIConfig()
av_api_client = AV_APIClient()
plot_component = Plot_Components()

# temporary value for API KEY
st.session_state["api_key"] = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
# AV_API_KEY = "WGHKWKAR5TGFV4IC"

############# PAGE STARTS HERE #############

st.title("Financial Ratio Analysis")

# ticker = st.selectbox(
#     "Select ticker:",
#     api_config.get_ticker_options(),
#     key="selectbox-main"
# )

# st.markdown(f"You selected: :green[{ticker}]")


# df = av_api_client.get_company_overview(ticker, AV_API_KEY)

# st.write(df)

plot_component.draw_pe_ratio_component(api_client=api_client, api_config=api_config, API_KEY=st.session_state["api_key"])

# Important metrics
# Return on equity
# Debt levels
# profit margins
# price to earning
# price to book
# revenuew growth
# earning growth
# analyst ratings
# news sentiment
