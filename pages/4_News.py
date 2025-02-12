import streamlit as st
from streamlit_extras.grid import grid
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from transformers import pipeline
import plotly.express as px
import finnhub

# importing api clients
from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient, get_fmp_client
from services.alphavantage_api_client import AlphaVantage_APIClient, get_alphavantage_client

# importing plot components
from view.alphavantage_plot_components import AlphaVantage_Plot_Components
from view.finnhub_plot_components import Finnhub_Plot_Components
from view.fmp_plot_components import FMP_Plot_Components


st.set_page_config(
    page_title="Profile Page",
    layout="wide",
)

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

# Initialising global variables
if "fmp_api_key" not in st.session_state:
    st.session_state["fmp_api_key"] = ""

if "av_api_key" not in st.session_state:
    st.session_state["av_api_key"] = ""

if "finnhub_api_key" not in st.session_state:
    st.session_state["finnhub_api_key"] = ""

if "qwen_api_key" not in st.session_state:
    st.session_state["qwen_api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

# Initialising FinBERT pipeline
pipe = pipeline("text-classification", model="ProsusAI/finbert")

def extract_titles(news_list):
    return [news['title'] for news in news_list]


pipe = pipeline("text-classification", model="ProsusAI/finbert")

############# PAGE STARTS HERE #############

st.title("News")

col1, col2 = st.columns(2)

with col1 :
    selectedTicker = st.selectbox(
        "Select ticker:",
        api_config.get_ticker_options(),
        placeholder = "Choose an option",
        key="selectbox1"
    )

with col2:
    numberOfNews = st.slider(
        "Number of news",
        min_value = 5,
        max_value= 50,
        value = 15,
        step = 5,
        key = "numberOfNewsSlider"
    )

news = yf.Search(selectedTicker, news_count=numberOfNews).news

# Example news sentiment outputs
news_sentiments = pipe(extract_titles(news))

# Count the occurrences of each sentiment label
sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
for sentiment in news_sentiments: # pyright: ignore
    sentiment_counts[sentiment['label']] += 1 # pyright: ignore

df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count']) # pyright: ignore

# Define the custom color map
color_map = {
    'positive': '#00FF00',  # Green
    'neutral': '#FFEB3B',  # Yellow
    'negative': '#F44336'  # Red
}

# Mapping sentiment labels to numerical values
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}

# Weighted average of sentiment scores
weighted_scores = np.array([sentiment_map[item['label']] * item['score'] for item in news_sentiments]) # pyright: ignore
average_score = np.sum(weighted_scores) / np.sum([item['score'] for item in news_sentiments]) # pyright: ignore

# Determine the aggregated sentiment label
aggregated_sentiment = "positive" if average_score > 0 else "negative" if average_score < 0 else "neutral"

with st.container(border=True):
    st.header("Sentiment News Analysis")

    g = grid([0.4, 0.6], vertical_align="center")
    # Create the ring chart using Plotly Express
    fig = px.pie(df,
                names='Sentiment',
                values='Count',
                hole=0.4,  # Ring effect
                color='Sentiment',
                color_discrete_map=color_map)

    # Display the chart in Streamlit
    g.plotly_chart(fig)
    g.markdown(f"""
        ### Aggregated Sentiment: {aggregated_sentiment}
        ### Score: {average_score}

    """)

news_filter = st.selectbox(
    "Filter News Option",
    ['All','Positive', 'Neutral', 'Negative'],
    key="news_filter"
)

for article, sentiment in zip(news,news_sentiments): # pyright: ignore
    with st.container():
        g = grid([0.3, 0.7], vertical_align="center")

        # Handle image display
        def get_image(article):
            if ('thumbnail' in article and 'resolutions' in article['thumbnail']
                and article['thumbnail']['resolutions']
                and 'url' in article['thumbnail']['resolutions'][0]):
                return article['thumbnail']['resolutions'][0]['url']
            return "assets/default_news.jpeg"

        # Create markdown content
        sentiment_icons = {
            'positive': '🟢',
            'neutral': '🟡',
            'negative': '🔴',
        }

        def create_markdown(article, sentiment):
            return f"""
                ### {sentiment_icons[sentiment['label']]} [{article['title']}]({article['link']})
                Publisher: {article['publisher']} {datetime.datetime.fromtimestamp(article['providerPublishTime'])}
                | Sentiment Rating: {sentiment['label']}
            """

        # Display content if it matches filter
        if (news_filter.lower() == 'all' or
            news_filter.lower() == sentiment['label']):
            g.image(get_image(article))
            g.markdown(create_markdown(article, sentiment))
