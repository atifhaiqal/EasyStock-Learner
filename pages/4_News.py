import streamlit as st
from streamlit_extras.grid import grid
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from transformers import pipeline
import plotly.express as px
import finnhub
import google.generativeai as genai

# importing api clients
from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient, get_fmp_client
from services.alphavantage_api_client import AlphaVantage_APIClient, get_alphavantage_client
from services.alpaca_api_client import Alpaca_APIClient, get_alpaca_client
from config.api_settings import Alpaca_APIConfig

# importing plot components
from view.alphavantage_plot_components import AlphaVantage_Plot_Components
from view.finnhub_plot_components import Finnhub_Plot_Components
from view.fmp_plot_components import FMP_Plot_Components


st.set_page_config(
    page_title="News Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

#temp
FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'
FINNHUB_API_KEY = 'ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg'

fmp_api = get_fmp_client(FMP_API_KEY)
av_api = get_alphavantage_client(AV_API_KEY)
finnhub_client = finnhub.Client(FINNHUB_API_KEY)
alpaca_api = get_alpaca_client(Alpaca_APIConfig.get_alpaca_api_key, Alpaca_APIConfig.get_alpaca_secret_key)

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

if "selected_sentiment" not in st.session_state:
    st.session_state.selected_sentiment = "All"

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ['AAPL']

if "llm_summary" not in st.session_state:
    st.session_state.llm_summary = "No summary currently."

# Initialising FinBERT pipeline
pipe = pipeline("text-classification", model="ProsusAI/finbert")

def extract_titles(news_list):
    return [news['headline'] for news in news_list]


pipe = pipeline("text-classification", model="ProsusAI/finbert")

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
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="You are a an assistant in a investment learning app called EasyStock Learner. Your role is to predict the stock rating of a company given its financial data and give the reasoning behind the rating (either buy or sell). Apart from that your role is also to help users understand the meaning behind the different metrics and ratios in finance. You are required to analyse time series financial data and generate an explanation of the performance of the metric. Be concise and start directly with insights. Explain in a tone thats easy to understand for someone with little to moderate financial litteracy.",
)

if "chat_session" not in st.session_state:
    st.session_state.chat_session = genai.ChatSession(model)

#---------------------------------------------------------------------------#
# Additional Functions
def get_image(article):
    if "images" in article and article["images"]:
        return article["images"][0]["url"]
    return "https://www.nccpimandtip.gov.eg/uploads/newsImages/1549208279-default-news.png"  # Default image

def format_date(iso_date):
    return datetime.datetime.fromisoformat(iso_date[:-1]).strftime("%Y-%m-%d %H:%M:%S")

def create_markdown(article, sentiment):
    return f"""
        #### {article['headline']}
        **Source**: {article['source']} | **Published**: {format_date(article['created_at'])} \n\n
        Sentiment: {sentiment_icons[sentiment['label']]} {sentiment['label'].capitalize()} \n\n
        [**Link**]({article['url']})
    """
def update_filter(trace, points, selector):
        if points.point_inds:  # If a slice was clicked
            selected_category = df.iloc[points.point_inds[0]]['Sentiment']
            st.session_state.selected_sentiment = selected_category  # Update session state

def get_labeled_alpaca_news(titles,news_sentiments):
    labeled_news = [(title, sentiment['label'], sentiment['score'])
                    for title, sentiment in zip(titles, news_sentiments)]

    return labeled_news

def make_summary(labeled_news):
    message = f"""
            You are provided with a list of news items. Each item includes text (like a headline or short description) and a sentiment label (e.g., 'Positive', 'Negative', 'Neutral') derived from FinBERT analysis.

            News Items:
            {labeled_news} 
            # Ensure labeled_news is formatted clearly, e.g., a list of strings like:
            # "- [Positive] Company X reports record profits."
            # "- [Negative] Regulator investigates Company Y."
            # Or a list of dictionaries represented as a string.

            Task:
            Please synthesize the information from the news items above into a cohesive paragraph. Your summary should:
            1. Identify the main themes, companies, or events discussed.
            2. Describe the overall sentiment surrounding these key themes, explicitly mentioning whether the news is generally positive, negative, or mixed based on the provided labels.
            3. Keep the summary concise and informative.
            4. Provide the summary in a markdown format.
            """

    chat_session = st.session_state.chat_session
    response = chat_session.send_message(message)

    return response.text

sentiment_icons = {
    'positive': '🟢',
    'neutral': '🟡',
    'negative': '🔴',
}

############# PAGE STARTS HERE #############

with st.sidebar:
    st.title(":green[EasyStock] Learner :chart:")
    
    st.session_state.selected_tickers = st.multiselect(
        "Select ticker:",
        api_config.get_ticker_options().keys(),
        default=st.session_state.selected_tickers,
        key="sidebar_selectbox",
        format_func=lambda x: api_config.get_ticker_options()[x],
        max_selections=3
    )       
    
    with st.container(border=True):
        st.header("Links to other pages")
        st.page_link("tutorial.py", label="Tutorial")
        st.page_link("pages/1_🏠_Homepage.py", label="Dashboard")
        st.page_link("pages/2_Financial Ratio Analysis.py", label="Assisted Analysis")
        st.page_link("pages/6_About.py", label="About")

st.title("News")

main_col, right_col = st.columns((6,3), gap='medium')

with main_col:

    col = st.columns(6)

    with col[0]:
        selectedTicker = st.selectbox(
            "Select ticker:",
            ['All'] + st.session_state.selected_tickers, 
            placeholder="Choose an option",
            key="ticker_selectbox"
        )

    with col[1]:
        news_filter = st.selectbox(
            "Filter News Option",
            ['All','Positive', 'Neutral', 'Negative'],
            key="news_filter"
        )


    with col[2]:
        dt_start = st.date_input(
            "Start Date",
            value = datetime.datetime(2024, 1, 3, 0, 0, 0),
            key = 'start_date_select',
        )

    with col[3]:
        dt_end = st.date_input(
            "End Date",
            value = "today",
            key = 'end_date_select',
        )

    with col[4]:
        news_order = st.selectbox(
            "News order",
            ['Descending','Ascending'],
            key="news_order"
        )

    with col[5]:
        numberOfNews = st.slider(
            "Number of news",
            min_value = 5,
            max_value= 50,
            value = 15,
            step = 5,
            key = "numberOfNewsSlider"
        )

    # news = yf.Search(selectedTicker, news_count=numberOfNews).news
    # dt_start = datetime.datetime(2024, 1, 3, 0, 0, 0)
    # dt_end = datetime.datetime(2025, 3, 3, 0, 0, 0)
    if selectedTicker == 'All':
        alpaca_news = alpaca_api.get_news(st.session_state.selected_tickers, alpaca_api.get_alpaca_datetime(dt_start), alpaca_api.get_alpaca_datetime(dt_end), limit=numberOfNews)
    else:
        alpaca_news = alpaca_api.get_news(selectedTicker, alpaca_api.get_alpaca_datetime(dt_start), alpaca_api.get_alpaca_datetime(dt_end), limit=numberOfNews)

    if alpaca_news is not None:
        news = alpaca_news
    else:
        news = []

    # Example news sentiment outputs
    titles = extract_titles(news)
    news_sentiments = pipe(titles) if titles else []
    labeled_news = get_labeled_alpaca_news(titles, news_sentiments)

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

    if(news_order == 'Ascending'):
        news = news[::-1]

    for i, (article, sentiment) in enumerate(zip(news, news_sentiments)):  # pyright: ignore

        # Filter articles based on selected sentiment
        if (news_filter.lower() == 'all' or news_filter.lower() == sentiment['label']):

            with st.container(border=True):
                g = st.columns((1,6))  # Define layout columns (image, text)

                # Display image
                with g[0]:
                    st.markdown(f"""
                    <a href="{article['url']}" target="_blank">
                        <img src="{get_image(article)}" style="width: 100%; height: 150; border-radius: 10px;">
                    </a>
                    """, unsafe_allow_html=True)

                # Display text content
                with g[1]:
                    st.markdown(create_markdown(article, sentiment))

with right_col:
    
    with st.container(border=False):
        st.header("Sentiment News Analysis")

        # Create the ring chart using Plotly Express
        fig = px.pie(df,
                    names='Sentiment',
                    values='Count',
                    hole=0.4,  # Ring effect
                    color='Sentiment',
                    color_discrete_map=color_map)

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  
            height=250,  
            width=250, 
        )

        st.plotly_chart(fig)

        sentiment = aggregated_sentiment.capitalize()

        if(sentiment == 'Positive'):
            st.markdown(f"Aggregated Sentiment: :green[{sentiment}] ")
        elif(sentiment == 'Neutral'):
            st.markdown(f"Aggregated Sentiment: :yellow[{sentiment}] ")
        elif(sentiment == 'Negative'):
            st.markdown(f"Aggregated Sentiment: :red[{sentiment}] ")

    if st.button("Summarise news"):
            with st.spinner("Making prediction... Please wait", show_time=True):
                    st.session_state.llm_summary = make_summary(labeled_news)

    st.markdown(st.session_state.llm_summary)


