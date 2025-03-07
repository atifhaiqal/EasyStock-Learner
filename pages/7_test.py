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
from services.qwen_api_client import Qwen_APIClient
from config.api_settings import YF_APIConfig
from services.alphavantage_api_client import AlphaVantage_APIClient
import finnhub
import os
from openai import OpenAI
from config.api_settings import Qwen_LLM_APIConfig
import datetime


# Page config
st.set_page_config(
    page_title="TEST",
)

FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'

api_config = YF_APIConfig()
qwen_api_key = Qwen_LLM_APIConfig().get_qwen_api_key()[0]
qwen_client = Qwen_APIClient.get_qwen_client(qwen_api_key)
fmp_api = FMP_APIClient(FMP_API_KEY)
av_api = AlphaVantage_APIClient(AV_API_KEY)
fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()
av_plot = AlphaVantage_Plot_Components()


############# PAGE STARTS HERE #############

st.title("TESTING PAGE")

# finnhub_client = finnhub.Client(api_key="ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg")

# selectedTickers = st.multiselect(
#     "Select ticker:",
#     api_config.get_ticker_options(),
#     key="selectbox1"
# )

# combined_df = fin_plot.draw_pe_ratio(selectedTickers, finnhub_client)

# df = pd.DataFrame(combined_df)

# if st.button("Give me insights!"):
#     try:
#         qwen_client.describe_pe_ratio(df)
#     except Exception as e:
#         print(f"Error message: {e}")
#         print("For more information, see: https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")

# metrics = finnhub_client.company_basic_financials('AAPL', 'all')
# st.write(metrics)

# metrics = finnhub_client.company_basic_financials('AAPL', 'all')
# dividendYield = metrics["metric"]["dividendYieldIndicatedAnnual"]
# st.write(dividendYield)

# fin_plot.draw_dividend_yield_annual(selectedTickers, finnhub_client)

# incomeStatement = fmp_api.get_income_statement("AAPL")

# st.write(incomeStatement)

# av_plot.draw_stock_prices(selectedTickers, av_api)

# url = "https://data.alpaca.markets/v1beta1/news?start=2024-01-03T00%3A00%3A00Z&end=2025-03-03T00%3A00%3A00Z&sort=desc&symbols=AAPL&limit=10"
# headers = {
#     "accept": "application/json",
#     "APCA-API-KEY-ID": "PKWSKNYCB90Y48I8T40S",
#     "APCA-API-SECRET-KEY": "sbAZegA51WHldKlV6hLkHtgRepXDy8Yq7txtTtLm"
# }

# response = requests.get(url, headers=headers)

# print(response.json())

# alpaca_news = response.json()

from services.alpaca_api_client import Alpaca_APIClient, get_alpaca_client
from config.api_settings import Alpaca_APIConfig

api = get_alpaca_client(Alpaca_APIConfig.get_alpaca_api_key, Alpaca_APIConfig.get_alpaca_secret_key)

dt_start = datetime.datetime(2024, 1, 3, 0, 0, 0)
dt_end = datetime.datetime(2025, 3, 3, 0, 0, 0)
alpaca_news = api.get_news("AAPL", api.get_alpaca_datetime(dt_start), api.get_alpaca_datetime(dt_end))

# alpaca_news = {  # Replace this with your actual API response
#     "news": [
#         {
#             "author": "Benzinga Newsdesk",
#             "created_at": "2025-03-04T12:31:54Z",
#             "headline": "Repligen Purchases 908 Devices' Desktop Portfolio Of Four Devices...",
#             "source": "benzinga",
#             "url": "https://www.benzinga.com/m-a/25/03/44105476/repligen-purchases-908...",
#             "images": [],
#         },
#         {
#             "author": "Murtuza Merchant",
#             "created_at": "2025-03-04T12:31:11Z",
#             "headline": "Bitcoin Bottom Will Be $70,000 'At Worst,' Arthur Hayes Says",
#             "source": "benzinga",
#             "url": "https://www.benzinga.com/markets/cryptocurrency/25/03/44105461/bitcoin-bottom-will...",
#             "images": [
#                 {
#                     "size": "large",
#                     "url": "https://cdn.benzinga.com/files/imagecache/2048x1536xUP/images/story/2025/03/04/bitcoin-ai.png"
#                 }
#             ],
#         }
#     ]
# }

if alpaca_news is not None:
    news = alpaca_news
else:
    news = []

# Display each article
for article in news:
    with st.container(border=True):

        g = st.columns([0.3, 0.7])  # Define layout columns (image, text)

        # Handle image display
        def get_image(article):
            if "images" in article and article["images"]:
                return article["images"][0]["url"]  # Use the first image if available
            return "https://www.nccpimandtip.gov.eg/uploads/newsImages/1549208279-default-news.png"  # Default image

        # Format date
        def format_date(iso_date):
            return datetime.datetime.fromisoformat(iso_date[:-1]).strftime("%Y-%m-%d %H:%M:%S")

        # Create markdown content
        def create_markdown(article):
            return f"""
                ### {article['headline']}
                **Source**: {article['source']} | **Published**: {format_date(article['created_at'])}
                [**Link**]({article['url']})
            """

        # Display content
        with g[0]:  # Image column
            st.markdown(
                f"""
                <a href="{article['url']}" target="_blank">
                    <img src="{get_image(article)}" style="width: 100%; height: auto; border-radius: 10px;">
                </a>
                """,
                unsafe_allow_html=True
            )
            # st.image(get_image(article))

        with g[1]:  # Text column
            st.markdown(create_markdown(article))
