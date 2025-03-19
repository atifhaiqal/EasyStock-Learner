import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date
import finnhub
from transformers import pipeline
import datetime
from streamlit_extras.metric_cards import style_metric_cards
import requests
import google.generativeai as genai
import os

# importing api clients
from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient, get_fmp_client
from services.alphavantage_api_client import AlphaVantage_APIClient, get_alphavantage_client
from services.finnhub_api_client import Finnhub_APIClient, get_finnhub_client
from services.alpaca_api_client import Alpaca_APIClient, get_alpaca_client
from config.api_settings import Alpaca_APIConfig, Qwen_LLM_APIConfig
# from services.gemini_api_client import Gemini_APIClient, get_gemini_client

# importing plot components
from view.alphavantage_plot_components import AlphaVantage_Plot_Components
from view.finnhub_plot_components import Finnhub_Plot_Components
from view.fmp_plot_components import FMP_Plot_Components
from view.yfinance_plot_componenets import YFinance_Plot_Components


############# PAGE CONFIG #############
st.set_page_config(
    page_title="EasyStock Learner",
    page_icon= "💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = "" #temporary for testing purposes

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ['AAPL']

#temp
FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'
FINNHUB_API_KEY = 'ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg'

fmp_api = get_fmp_client(FMP_API_KEY)
av_api = get_alphavantage_client(AV_API_KEY)
finnhub_client = get_finnhub_client(FINNHUB_API_KEY)
api_config = FMP_APIConfig()
alpaca_api = get_alpaca_client(Alpaca_APIConfig.get_alpaca_api_key, Alpaca_APIConfig.get_alpaca_secret_key)
# gemini_api = get_gemini_client()

fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()
av_plot = AlphaVantage_Plot_Components()
y_plot = YFinance_Plot_Components()

pipe = pipeline("text-classification", model="ProsusAI/finbert")

# temporary value for API KEY
st.session_state["api_key"] = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"

sp500_financial_df = pd.read_csv("ml_models/constituents-financials.csv")

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

#---------------------------------------------------------------------------#
# Additional Functions

# Define the custom color map
color_map = {
    'positive': '#00FF00',  # Green
    'neutral': '#FFEB3B',  # Yellow
    'negative': '#F44336'  # Red
}

def get_image(article):
    if "images" in article and article["images"]:
        return article["images"][0]["url"]
    return "https://www.nccpimandtip.gov.eg/uploads/newsImages/1549208279-default-news.png"  # Default image

def format_date(iso_date):
    return datetime.datetime.fromisoformat(iso_date[:-1]).strftime("%Y-%m-%d %H:%M:%S")

def create_markdown(article):
    return f"""
        ###### {article['headline']}
        **Source**: {article['source']} | **Published**: {format_date(article['created_at'])} \n\n
        [**Link**]({article['url']})
    """

def extract_titles(news_list):
    return [news['headline'] for news in news_list]

@st.cache_resource
def news_sentiment_analyis(tickers):
    dt_start = datetime.datetime(2024, 1, 3, 0, 0, 0)
    dt_end = datetime.datetime.today()
    combined_df = pd.DataFrame()
    aggregated_sentiment_list = []

    for ticker in tickers:
        alpaca_news = alpaca_api.get_news(ticker, alpaca_api.get_alpaca_datetime(dt_start), alpaca_api.get_alpaca_datetime(dt_end), limit=20)

        if alpaca_news is not None:
            news = alpaca_news
        else:
            news = []

        # Example news sentiment outputs
        titles = extract_titles(news)
        news_sentiments = pipe(titles) if titles else []

        # Count the occurrences of each sentiment label
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        for sentiment in news_sentiments: # pyright: ignore
            sentiment_counts[sentiment['label']] += 1 # pyright: ignore

        list_items = list(sentiment_counts.items())
        df = pd.DataFrame(list_items, columns=['Sentiment', 'Count']) # pyright: ignore

        df['ticker'] = ticker # Add column to identify ticker
        combined_df = pd.concat([combined_df, df])

        # Mapping sentiment labels to numerical values
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}

        # Weighted average of sentiment scores
        weighted_scores = np.array([sentiment_map[item['label']] * item['score'] for item in news_sentiments]) # pyright: ignore
        average_score = np.sum(weighted_scores) / np.sum([item['score'] for item in news_sentiments]) # pyright: ignore

        # Determine the aggregated sentiment label
        aggregated_sentiment = "positive" if average_score > 0 else "negative" if average_score < 0 else "neutral"

        aggregated_sentiment_list.append((ticker, aggregated_sentiment))

    aggregated_sentiment_df = pd.DataFrame(aggregated_sentiment_list, columns=['ticker','average_sentiment'])
    combined_df = combined_df.reset_index(drop=True)

    return combined_df, aggregated_sentiment_df

def format_number(num):
    if num >= 1_000_000_000_000:  # Trillions
        return f"{num / 1_000_000_000_000:.1f}T"
    elif num >= 1_000_000_000:  # Billions
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:  # Millions
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:  # Thousands
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def calc_percentage_difference(current_val, previous_val):
    difference = current_val - previous_val
    percentage_difference = difference/previous_val
    percentage = f"{percentage_difference:.2%}"
    return percentage

def make_diff_fmp_cf(ticker, period, label, metric):
    try:
        cashflow_df = fmp_api.get_cashflow_statement(ticker)
        
        current_year = cashflow_df.iloc[0][metric]

        if period == 1:
            prev_year = cashflow_df.iloc[period][metric]
        else:
            prev_year = cashflow_df.iloc[period-1][metric]

        difference = calc_percentage_difference(current_year, prev_year)

        st.metric(label=label, value=format_number(current_year), delta=difference)
    except:
        with st.container(border=True):
            st.markdown(f"{label} is unavailable for {ticker}")

def make_diff_fmp_bs(ticker, period, label, metric):
    try:
        balancesheet_df = fmp_api.get_balance_sheet(ticker)
        
        current_year = balancesheet_df.iloc[0][metric]

        if period == 1:
            prev_year = balancesheet_df.iloc[period][metric]
        else:
            prev_year = balancesheet_df.iloc[period-1][metric]

        difference = calc_percentage_difference(current_year, prev_year)

        st.metric(label=label, value=format_number(current_year), delta=difference)
    except:
        with st.container(border=True):
            st.markdown(f"{label} is unavailable for {ticker}")

def make_diff_fmp_is(ticker, period, label, metric):
    try: 
        incomeStatement_df = fmp_api.get_income_statement(ticker)
        
        current_year = incomeStatement_df.iloc[0][metric]

        if period == 1:
            prev_year = incomeStatement_df.iloc[period][metric]
        else:
            prev_year = incomeStatement_df.iloc[period-1][metric]

        difference = calc_percentage_difference(current_year, prev_year)

        st.metric(label=label, value=format_number(current_year), delta=difference)
    except:
        with st.container(border=True):
            st.markdown(f"{label} is unavailable for {ticker}")

def make_diff_finnhub(ticker, period, label, metric):
    try:
        metrics = finnhub_client.get_company_basic_financials(ticker, 'all')
        peRatio = metrics["series"]["annual"][metric]
        df = pd.json_normalize(peRatio)

        current_year = df.iloc[0]['v']

        if period == 1:
            prev_year = df.iloc[period]['v']
        else:
            prev_year = df.iloc[period-1]['v']

        difference = calc_percentage_difference(current_year, prev_year)

        st.metric(label=label, value=f"{current_year:.1f}", delta=difference)
    except:
        with st.container(border=True):
            st.markdown(f"{label} is unavailable for {ticker}")

def make_performance_chart(ticker):
    with st.container(border=True):
            st.markdown(f"### {ticker}")
            inner_col = st.columns(3)

            with inner_col[0]:
                make_diff_finnhub(ticker, period_selectbox, "PE Ratio", "pe")
                make_diff_finnhub(ticker, period_selectbox, "PTBV", "ptbv")
                make_diff_fmp_is(ticker, period_selectbox, "EBITDA", "ebitda")
                make_diff_finnhub(ticker, period_selectbox, "Dept to Capital", "totalDebtToTotalCapital")
                
            with inner_col[1]:
                make_diff_finnhub(ticker, period_selectbox, "PB Ratio", "pb")
                make_diff_finnhub(ticker, period_selectbox, "Return on Assets", "roa")
                make_diff_fmp_is(ticker, period_selectbox, "Revenue", "revenue")
                make_diff_fmp_bs(ticker, period_selectbox, "Total Debt", "totalDebt")

            with inner_col[2]:
                make_diff_finnhub(ticker, period_selectbox, "Earning per Share", "eps")
                make_diff_finnhub(ticker, period_selectbox, "Return on Equity", "roe")
                make_diff_fmp_is(ticker, period_selectbox, "Net Income", "netIncome")
                make_diff_fmp_cf(ticker, period_selectbox, "Free Cash Flow", "freeCashFlow")

def make_prediction(financial_data, news_sentiment):
    financial_data_json = financial_data.to_json(orient="records", indent=4)
    if news_sentiment is not None:
        message = f"""Here is the financial data of a given company:\n\n{financial_data_json}\n\n.
                    The sentiment analysis of 20 of the most recent news surrounding the company is {news_sentiment}.
                    """
    else:
        message = f"""Here is the financial data of a given company:\n\n{financial_data_json}\n\n.
                    """
    chat_session = st.session_state.chat_session
    response = chat_session.send_message(f"""{message} Given these data points, make a stock rating on the given company (buy, hold, or sell). 
                    Analyse the performance and provide a concise insights but avoid introductions.
                    Highlight key reasonings as to why the decision has been made. 
                    Give the results in markdown format, with headers for key reasons for the rating of the stock. 

                    Return the rating as a markdown header in the form of "## Rating: :green[Buy]", "## Rating: :orange[Hold]" or "## Rating: :red[Sell]",
                    depending on the stock rating. This is to comply with the formatting of the UI.""")

    return response.text

############# PAGE STARTS HERE #############

with st.sidebar:
    st.title(":green[EasyStock] Learner :chart:")
    
    st.session_state.selected_tickers = st.multiselect(
        "Select ticker:",
        api_config.get_ticker_options().keys(),
        default=st.session_state.selected_tickers,
        key="selectbox1",
        format_func=lambda x: api_config.get_ticker_options()[x],
        max_selections=3
    )       

    st.write("**Price and volumne chart**")

    show_candlestick = st.toggle("Show candlestick", 
                                   value=True,
                                   help='''
                                        A candlestick chart helps visualize stock price movements over time.
                                        Each candlestick represents a specific time period (e.g., 1 day) and shows four key prices:

                                        - Open – The price when the period started
                                        - Close – The price when the period ended
                                        - High – The highest price reached
                                        - Low – The lowest price reached
                                        ''',
                                )
    
    show_volume = st.toggle("Show volume", 
                                   value=True,
                                   help='''
                                        Volume shows the trading activity behind price movements:

                                        Quantity – The total number of shares traded during the period
                                        Confirmation – High volume reinforces the significance of price changes (e.g., a price rise with high volume suggests strong buyer conviction)
                                        Divergence – Declining volume during a trend may signal weakening momentum
                                        This complements candlestick patterns by revealing whether market participants strongly support the price action or if movements lack conviction.
                                        ''',
                                )

    st.write("**News sentiment chart**")

    invert_sunburst = st.toggle("Invert news sentiment chart",
                                    value=False,
                                    help='''
                                        The news sentiment is shown using a sunburst chart. 
                                        A sunburst chart is a type of visualization used to display hierarchical data in a circular format. 
                                        Each layer of the chart represents a level in the hierarchy with the size of each segment corresponding to the value of that category, 
                                        and different colors used to distinguish between them.

                                        Sunburst charts are helpful for showing the relationships between parts of a whole, 
                                        making it easy to identify patterns and distributions within complex hierarchical data.

                                        Try inverting it to see other patterns
                                        '''
                                )
    
    with st.container(border=True):
        st.header("Links to other pages")
        st.page_link("pages/2_Financial Ratio Analysis.py", label="Assisted Analysis")
        st.page_link("pages/4_News.py", label="News Analysis")
        st.page_link("pages/6_About.py", label="About")
        st.page_link("pages/7_test.py", label="Test")

st.title(":green[Dashboard]")

col = st.columns((1.5, 4.5, 2), gap='small')

with col[0]:
    fin_plot.draw_consensus_ratings(st.session_state.selected_tickers, finnhub_client)
    fin_plot.draw_stock_ratings(st.session_state.selected_tickers, finnhub_client)
with col[1]:
    # st.markdown("### Price and Volume Analysis")
    av_plot.draw_combined_price_volume_chart(st.session_state.selected_tickers, show_candlestick, show_volume, av_api)
with col[2]:
    df, aggregated_sentiment_df  = news_sentiment_analyis(st.session_state.selected_tickers)

    st.markdown("### News Sentiment Breakdown")

    color_map = {
        'positive': '#3BD133',  
        'neutral': '#FBF909',   
        'negative': '#FB1509'   
    }

    news_fig = px.sunburst(
        df, 
        names='ticker',
        path=['ticker', 'Sentiment'], 
        values='Count',
        color='Sentiment',  # Assign colors based on Sentiment
        color_discrete_map=color_map  # Custom color map
    )

    news_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),  
        height=250,  
        width=250,  
    )

    news_fig_invert = px.sunburst(
        df, 
        names='ticker',
        path=['Sentiment', 'ticker'], 
        values='Count',
        color='Sentiment',  # Assign colors based on Sentiment
        color_discrete_map=color_map  # Custom color map
    )

    news_fig_invert.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),  
        height=250,  
        width=250, 
    )
    
    # Display the chart in Streamlit
    if invert_sunburst:
        st.plotly_chart(news_fig_invert)
    else:
        st.plotly_chart(news_fig)

    st.markdown("#### Average News Sentiments")

    for ticker in st.session_state.selected_tickers:
        sentiment = aggregated_sentiment_df.loc[aggregated_sentiment_df['ticker'] == ticker, 'average_sentiment'].values[0].capitalize()

        if(sentiment == 'Positive'):
            st.markdown(f"#### {ticker}: :green[{sentiment}] ")
        elif(sentiment == 'Neutral'):
            st.markdown(f"#### {ticker}: :yellow[{sentiment}] ")
        elif(sentiment == 'Negative'):
            st.markdown(f"#### {ticker}: :red[{sentiment}] ")

col = st.columns((6, 2), gap='small')

with col[0]:
    with st.container(border=False, height=716):
        st.header('Predictive Rating')

        inner_col = st.columns(4, vertical_alignment="bottom")

        with inner_col[0]:
            predictive_selectbox = st.selectbox(
                "Select a ticker to rate:",
                st.session_state.selected_tickers,
                index=0,
                key="predictive_selectbox"
            )

        with inner_col[1]:
            include_news_selectbox = st.selectbox(
                "Include news?:",
                ['Yes', 'No'],
                index=0,
                key="include_news_selectbox"
            )

        with inner_col[2]:
            selected_ticker_df = sp500_financial_df.loc[sp500_financial_df['Symbol'] == predictive_selectbox]

            finaical_df = selected_ticker_df.drop(columns=["Symbol", "Name", "SEC Filings"])
            news_sentiment = aggregated_sentiment_df.loc[aggregated_sentiment_df['ticker'] == predictive_selectbox, 'average_sentiment'].values[0].capitalize()

            write_analysis = False
            make_rating = False
            if st.button("Make prediction"):
                with inner_col[3]:  # Place spinner in the next column
                    with st.spinner("Making prediction... Please wait", show_time=True):
                        if include_news_selectbox == "Yes":
                            llm_analysis = make_prediction(finaical_df, news_sentiment)
                        else:
                            llm_analysis = make_prediction(finaical_df, None)
                        write_analysis = True

        # st.dataframe(finaical_df)
        # st.write(news_sentiment)

        if write_analysis:
            # st.write(llm_analysis)          
            st.markdown(f"{llm_analysis}")

with col[1]:
    with st.container(border=True):
        news_selectbox = st.selectbox(
            "Select ticker:",
            st.session_state.selected_tickers,
            index=0,
            key="news_selectbox"
        )
    
        with st.container(border=False, height=600):
            dt_start = datetime.datetime(2024, 1, 3, 0, 0, 0)
            dt_end = datetime.datetime.today()
            alpaca_news = alpaca_api.get_news(news_selectbox, alpaca_api.get_alpaca_datetime(dt_start), alpaca_api.get_alpaca_datetime(dt_end), limit=5)

            if alpaca_news is not None:
                news = alpaca_news
            else:
                news = []

            for i, article in enumerate(news):  # pyright: ignore
            
                with st.container(border=True): 
                    g = st.columns([0.3, 0.7])  # Define layout columns (image, text)

                    # Display image
                    with g[0]:
                        st.markdown(f"""
                        <a href="{article['url']}" target="_blank">
                            <img src="{get_image(article)}" style="width: 100%; height: auto; border-radius: 10px;">
                        </a>
                        """, unsafe_allow_html=True)

                    # Display text content
                    with g[1]:
                        st.markdown(create_markdown(article))

            st.page_link("pages/4_News.py", label="More News")

with st.container(border=True):
    col_l, col_r = st.columns((7,1), gap='small')

    with col_l:
        st.header("Stock Perfomance Overview")

    with col_r:
        period_selectbox = st.selectbox(
                "Select time period (Years):",
                options=[1,5],
                index=0,
                key="period_selectbox"
            )

    col = st.columns(len(st.session_state.selected_tickers))

    for i, ticker in enumerate(st.session_state.selected_tickers):
        with col[i]:
            make_performance_chart(ticker)
    
    st.page_link("pages/2_Financial Ratio Analysis.py", label="Further Analysis and Visualisation")

    style_metric_cards(background_color='#2C2C2C', border_color='#1C1C1C', border_left_color='#1C1C1C')
