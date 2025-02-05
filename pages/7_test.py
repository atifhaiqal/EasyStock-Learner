# THIS IS PURELY FOR TEST PURPOSES, DO NOT USE IN PRODUCTION

from numpy.lib.function_base import diff
from streamlit.elements import metric
import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from view import finnhub_plot_components
from view.finnhub_plot_components import Finnhub_Plot_Components
from view.alphavantage_plot_components import AlphaVantage_Plot_Components
from view.fmp_plot_components import FMP_Plot_Components
from services.fmp_api_client import FMP_APIClient
from config.api_settings import YF_APIConfig
from services.alphavantage_api_client import AlphaVantage_APIClient
import finnhub
import os
from openai import OpenAI
from config.api_settings import Qwen_LLM_APIConfig


# Page config
st.set_page_config(
    page_title="TEST",
)

FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'

api_config = YF_APIConfig()
qwen_api_key = Qwen_LLM_APIConfig().get_qwen_api_key()[0]
fmp_api = FMP_APIClient(FMP_API_KEY)
av_api = AlphaVantage_APIClient(AV_API_KEY)
fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()
av_plot = AlphaVantage_Plot_Components()


############# PAGE STARTS HERE #############

st.title("TESTING PAGE")

finnhub_client = finnhub.Client(api_key="ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg")

selectedTickers = st.multiselect(
    "Select ticker:",
    api_config.get_ticker_options(),
    key="selectbox1"
)

combined_df = fin_plot.draw_pe_ratio(selectedTickers, finnhub_client)

df = pd.DataFrame(combined_df)

# Convert DataFrame to JSON
pe_data_json = df.to_json(orient="records", indent=4)

if st.button("Give me insights!"):
    try:
        client = OpenAI(
            # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
            api_key= qwen_api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': 'You are a an assistant in a investment learning app called EasyStock Learner. Your role is to help interpret the importance table of an XGBoost model that predicts the stock rating of any given stock and give the reasoning behind the rating (either buy or sell). Apart from that your role is also to help users understand the meaning behind the different metrics and ratios in finance. You are required to analyse time series financial data and generate an explanation of the performance of the metric. Be concise and start directly with insights.'},
                {'role': 'user', 'content': f"Here is a list of companies and their P/E ratios:\n\n{pe_data_json}\n\nCan you analyze the performance and provide a concise insights but avoid introductions.?"}
                ]
        )
        print(completion.choices[0].message.content)
        st.write(completion.choices[0].message.content)
    except Exception as e:
        print(f"Error message: {e}")
        print("For more information, see: https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")

# metrics = finnhub_client.company_basic_financials('AAPL', 'all')
# st.write(metrics)

# metrics = finnhub_client.company_basic_financials('AAPL', 'all')
# dividendYield = metrics["metric"]["dividendYieldIndicatedAnnual"]
# st.write(dividendYield)

# fin_plot.draw_dividend_yield_annual(selectedTickers, finnhub_client)

# incomeStatement = fmp_api.get_income_statement("AAPL")

# st.write(incomeStatement)

# av_plot.draw_stock_prices(selectedTickers, av_api)
