import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date
import finnhub
from transformers import pipeline
import datetime
from streamlit_extras.metric_cards import style_metric_cards

# importing api clients
from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient, get_fmp_client
from services.alphavantage_api_client import AlphaVantage_APIClient, get_alphavantage_client
from services.finnhub_api_client import Finnhub_APIClient, get_finnhub_client
from services.alpaca_api_client import Alpaca_APIClient, get_alpaca_client
from config.api_settings import Alpaca_APIConfig

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

#temp
FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'
FINNHUB_API_KEY = 'ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg'

fmp_api = get_fmp_client(FMP_API_KEY)
av_api = get_alphavantage_client(AV_API_KEY)
finnhub_client = get_finnhub_client(FINNHUB_API_KEY)
api_config = FMP_APIConfig()
alpaca_api = get_alpaca_client(Alpaca_APIConfig.get_alpaca_api_key, Alpaca_APIConfig.get_alpaca_secret_key)

fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()
av_plot = AlphaVantage_Plot_Components()
y_plot = YFinance_Plot_Components()

pipe = pipeline("text-classification", model="ProsusAI/finbert")

# temporary value for API KEY
st.session_state["api_key"] = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"

#---------------------------------------------------------------------------#
# Additional Functions

# Define the custom color map
color_map = {
    'positive': '#00FF00',  # Green
    'neutral': '#FFEB3B',  # Yellow
    'negative': '#F44336'  # Red
}

def extract_titles(news_list):
    return [news['headline'] for news in news_list]

@st.cache_resource
def news_sentiment_analyis(tickers):
    dt_start = datetime.datetime(2024, 1, 3, 0, 0, 0)
    dt_end = datetime.datetime(2025, 3, 3, 0, 0, 0)
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

############# PAGE STARTS HERE #############

with st.sidebar:
    st.title(":green[EasyStock] Learner :chart:")
    
    selectedTickers = st.multiselect(
        "Select ticker:",
        api_config.get_ticker_options().keys(),
        default=['AAPL'],
        key="selectbox1",
        format_func=lambda x: api_config.get_ticker_options()[x],
        max_selections=3
    )       
    
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
        st.page_link("pages/4_News.py", label="News Analysis")
        st.page_link("pages/6_About.py", label="About")

st.title(":green[Dashboard]")

col = st.columns((1.5, 4.5, 2), gap='small')

with col[0]:
    fin_plot.draw_consensus_ratings(selectedTickers, finnhub_client)
    fin_plot.draw_stock_ratings(selectedTickers, finnhub_client)
with col[1]:
    av_plot.draw_stock_prices(selectedTickers, show_candlestick, av_api)
    av_plot.draw_volume_with_moving_average(selectedTickers, av_api)
with col[2]:
    df, aggregated_sentiment_df  = news_sentiment_analyis(selectedTickers)

    st.markdown("### News Sentiment Breakdown")

    color_map = {
        'positive': 'green',  
        'neutral': 'yellow',   
        'negative': 'red'   
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

    for ticker in selectedTickers:
        sentiment = aggregated_sentiment_df.loc[aggregated_sentiment_df['ticker'] == ticker, 'average_sentiment'].values[0].capitalize()

        if(sentiment == 'Positive'):
            st.markdown(f"#### {ticker}: :green[{sentiment}] ")
        elif(sentiment == 'Neutral'):
            st.markdown(f"#### {ticker}: :yellow[{sentiment}] ")
        elif(sentiment == 'Negative'):
            st.markdown(f"#### {ticker}: :red[{sentiment}] ")

st.header("Stock Perfomance Breakdown")

col = st.columns(3)

with col[0]:
    with st.container(border=True):
        st.markdown("### Stock A")
        inner_col = st.columns(2)

        with inner_col[0]:
            st.metric(label="PE Ratio", value=24.5, delta='5.5%')
            st.metric(label="PE Ratio", value=24.5, delta='-5.5%')
            st.metric(label="PE Ratio", value=24.5, delta='5.5%')

        with inner_col[1]:
            st.metric(label="Gain", value=5000, delta=1000)
            st.metric(label="PE Ratio", value=24.5, delta='-5.5%')
            st.metric(label="PE Ratio", value=24.5, delta='5.5%')

with col[1]:
    with st.container(border=True):
        st.markdown("### Stock B")
        inner_col = st.columns(2)

        with inner_col[0]:
            st.metric(label="PE Ratio", value=24.5, delta='15.5%')
            st.metric(label="PE Ratio", value=24.5, delta='-5.5%')
            st.metric(label="PE Ratio", value=24.5, delta='5.5%')

        with inner_col[1]:
            st.metric(label="Gain", value=5000, delta=1000)
            st.metric(label="PE Ratio", value=24.5, delta='-5.5%')
            st.metric(label="PE Ratio", value=24.5, delta='5.5%')

with col[2]:
    with st.container(border=True):
        st.markdown("### Stock C")
        inner_col = st.columns(2)

        with inner_col[0]:
            st.metric(label="PE Ratio", value=24.5, delta='5.5%')
            st.metric(label="PE Ratio", value=24.5, delta='-5.5%')
            st.metric(label="PE Ratio", value=24.5, delta='5.5%')

        with inner_col[1]:
            st.metric(label="Gain", value=5000, delta=-1000)
            st.metric(label="PE Ratio", value=24.5, delta='-5.5%')
            st.metric(label="PE Ratio", value=24.5, delta='5.5%')

style_metric_cards(background_color='#2C2C2C', border_color='#1C1C1C', border_left_color='#1C1C1C')

# if st.session_state["user_name"] and "" or st.session_state["api_key"] != "":
#     st.write("## Hi ", st.session_state["user_name"], "!")
# else:
#     st.write("## Please proceed to the profile page to set up API key and nickname")

# st.markdown("## Pick stocks to compare")

# selectedTickers = st.multiselect(
#     "Select ticker:",
#     api_config.get_ticker_options(),
#     default=['MSFT', 'GOOGL'],
#     key="selectbox1"
# )

# The main candle chart
# col1, col2 = st.columns([2,1])

# with col1:
#     av_plot.draw_stock_prices(selectedTickers, av_api)

# with col2:
#     fin_plot.draw_stock_ratings(selectedTickers, finnhub_client)
#     fin_plot.draw_consensus_ratings(selectedTickers, finnhub_client)


# with st.popover("How to read candle stick chart? 🤔"):
#     st.markdown('''
#     A candlestick chart helps visualize stock price movements over time.
#     Each candlestick represents a specific time period (e.g., 1 day) and shows four key prices:
#     ''')
#     st.markdown('''
#     - Open – The price when the period started
#     - Close – The price when the period ended
#     - High – The highest price reached
#     - Low – The lowest price reached

#     Understanding Candlesticks:
#     - Green Candle: The price closed higher than it opened (bullish).
#     - Red Candle: The price closed lower than it opened (bearish).
#     - Wicks: Thin lines above and below the candle body show the highest and lowest prices.

#     Key Patterns to Watch:
#     - Long green candles → Strong buying pressure
#     - Long red candles → Strong selling pressure
#     - Doji (small body, long wicks) → Market indecision
#     - Hammer / Shooting Star → Potential trend reversal
#     ''')

#     st.markdown('''
#     Candlestick charts help identify trends, reversals, and market sentiment quickly.
#     Use them with other indicators for better analysis! 🚀
#     ''')

# # Smaller charts
# col3, col4, col5, col6 = st.columns(4)

# with col3:
#     fmp_plot.draw_revenue(selectedTickers, fmp_api)
#     fin_plot.draw_pe_ratio(selectedTickers, finnhub_client)
#     fmp_plot.draw_total_debt(selectedTickers, fmp_api)

# with col4:
#     fmp_plot.draw_ebitda(selectedTickers, fmp_api)
#     fin_plot.draw_pb_ratio(selectedTickers, finnhub_client)
#     fmp_plot.draw_net_debt(selectedTickers, fmp_api)

# with col5:
#     fin_plot.draw_dividend_yield_annual(selectedTickers, finnhub_client)
#     fmp_plot.draw_longterm_debt(selectedTickers, fmp_api)
#     fmp_plot.draw_net_change_in_cash(selectedTickers, fmp_api)

# with col6:
#     fin_plot.draw_eps_ratio(selectedTickers, finnhub_client)
#     fmp_plot.draw_net_income(selectedTickers, fmp_api)
#     fmp_plot.draw_net_income_ratio(selectedTickers, fmp_api)

# st.header("Stock Rating Prediction (PLACEHOLDERS FOR NOW)")

# # use the same selection as the one on top

# st.subheader("AAPL")
# st.write("Stock Rating: :green[Buy]")
# st.write("Date: ", date.today())

# st.markdown('''
#     ### **Reasoning for Buy Call on Apple Stock (AAPL)**

#     #### **1. Strong Revenue Growth 📈**
#     - Apple’s latest earnings report shows a **15% YoY revenue increase**, driven by strong iPhone and services sales.
#     - Services segment (**App Store, iCloud, Apple Music**) is growing at **20% YoY**, providing stable recurring revenue.

#     #### **2. Positive Technical Indicators 📊**
#     - **50-day moving average** is crossing above the **200-day moving average** (**Golden Cross**), signaling an uptrend.
#     - Recent **candlestick patterns** show **higher lows**, indicating **bullish momentum**.

#     #### **3. Undervalued Relative to Growth 📉**
#     - Current **P/E ratio: 24** vs. historical average of **26**, suggesting slight undervaluation.
#     - **Price-to-book (P/B) ratio** remains stable, reflecting confidence in asset valuation.

#     #### **4. Strong Institutional Support 🏛️**
#     - **Hedge funds and institutional investors** have increased holdings by **8%** in the last quarter.
#     - **Warren Buffett’s Berkshire Hathaway** maintains a significant stake, showing long-term confidence.

#     #### **5. Macroeconomic & Industry Trends 🌍**
#     - The global shift toward **AI and AR** (**Apple Vision Pro**) positions Apple for future growth.
#     - **Declining inflation** and potential **Fed rate cuts** may boost tech stock valuations.
# ''')
