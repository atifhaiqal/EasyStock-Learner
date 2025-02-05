import streamlit as st
from config.api_settings import FMP_APIConfig
from services.alphavantage_api_client import AlphaVantage_APIClient
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

api_client = FMP_APIClient("OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw")
api_config = FMP_APIConfig()
av_api_client = AlphaVantage_APIClient("WGHKWKAR5TGFV4IC")
plot_component = Plot_Components()

# temporary value for API KEY
# st.session_state["api_key"] = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
# AV_API_KEY = "WGHKWKAR5TGFV4IC"

############# PAGE STARTS HERE #############

st.title("Financial Ratio Analysis")




# Important metrics
#
# Return on equity
# Debt levels
# profit margins
# price to earning /
# price to book /
# revenuew growth
# earning growth
# analyst ratings
# news sentiment
