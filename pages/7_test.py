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
# from services.gemini_api_client import Gemini_APIClient
from config.api_settings import YF_APIConfig
from services.alphavantage_api_client import AlphaVantage_APIClient
import finnhub
from openai import OpenAI
from config.api_settings import Qwen_LLM_APIConfig
import datetime
import google.generativeai as genai
import os

# Page config
st.set_page_config(
    page_title="TEST",
)

FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'

api_config = YF_APIConfig()
qwen_api_key = Qwen_LLM_APIConfig().get_qwen_api_key()[0]
# qwen_client = Gemini_APIClient.get_qwen_client(qwen_api_key)
fmp_api = FMP_APIClient(FMP_API_KEY)
av_api = AlphaVantage_APIClient(AV_API_KEY)
fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()
av_plot = AlphaVantage_Plot_Components()

genai.configure(api_key="AIzaSyDeEKkLTe_Gbv0jTn4Ormx5OUy8cuz8ahA")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
  system_instruction="You are a an assistant in a investment learning app called EasyStock Learner. Your role is to predict the stock rating of a company given its financial data and give the reasoning behind the rating (either buy or sell). Apart from that your role is also to help users understand the meaning behind the different metrics and ratios in finance. You are required to analyse time series financial data and generate an explanation of the performance of the metric. Be concise and start directly with insights.",
)

if "chat_session" not in st.session_state:
    st.session_state.chat_session = genai.ChatSession(model)

def make_prediction():
    chat_session = st.session_state.chat_session
    response = chat_session.send_message("""Given is the financial data of a company { "Sector": "Industrial Conglomerates", "Price": 152.2, "Price/Earnings": 21.286713, "Dividend Yield": 0.0199, "Earnings/Share": 7.15, "52 Week Low": 75.652176, "52 Week High": 155.0, "Market Cap": 83294183424, "EBITDA": 8117000192, "Price/Sales": 2.5520616, "Price/Book": 17.855467 }. The sentiment analysis on recent news is positive. Generate a stock rating on this company and provide reasons as to why that decision has been made. Be concise and avoid anything unrelated to the stock rating """)  

    return response.text

############# PAGE STARTS HERE #############

st.title("TESTING PAGE")

st.write(make_prediction())
