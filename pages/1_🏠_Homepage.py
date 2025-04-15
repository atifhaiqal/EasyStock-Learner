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
import csv
# import joblib

# importing api clients
from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient, get_fmp_client
from services.alphavantage_api_client import AlphaVantage_APIClient, get_alphavantage_client
from services.finnhub_api_client import Finnhub_APIClient, get_finnhub_client
from services.alpaca_api_client import Alpaca_APIClient, get_alpaca_client
from config.api_settings import Alpaca_APIConfig, Qwen_LLM_APIConfig

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

# Load the pipeline and label encoder
# pipeline = joblib.load('model_pipeline.joblib')
# label_encoder = joblib.load('label_encoder.joblib')

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
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="You are a an assistant in a investment learning app called EasyStock Learner. Your role is to predict the stock rating of a company given its financial data and give the reasoning behind the rating (either buy or sell). Apart from that your role is also to help users understand the meaning behind the different metrics and ratios in finance. You are required to analyse time series financial data and generate an explanation of the performance of the metric. Be concise and start directly with insights. Explain in a tone thats easy to understand for someone with little to moderate financial litteracy.",
)

if "chat_session" not in st.session_state:
    st.session_state.chat_session = genai.ChatSession(model)

if "llm_analysis" not in st.session_state:
    st.session_state.llm_analysis = "No predictions currently."


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

def extract_titles(news_list):
    return [news['headline'] for news in news_list]

def get_labeled_news(ticker):
    dt_start = datetime.datetime(2024, 1, 3, 0, 0, 0)
    dt_end = datetime.datetime.today()
    alpaca_news = alpaca_api.get_news(ticker, alpaca_api.get_alpaca_datetime(dt_start), alpaca_api.get_alpaca_datetime(dt_end), limit=20)

    if alpaca_news is not None:
        news = alpaca_news
    else:
        news = []

    # Example news sentiment outputs
    titles = extract_titles(news)
    news_sentiments = pipe(titles) if titles else []

    labeled_news = [(title, sentiment['label'], sentiment['score'])
                    for title, sentiment in zip(titles, news_sentiments)]

    return labeled_news

def format_financial_data(data, ticker):
    """
    Formats the raw financial data dictionary into a readable string for the LLM.
    Selects a curated set of key metrics relevant for stock rating analysis,
    removing redundancy and less critical data points to improve LLM focus.
    """
    
    metric_map = {
        # === Context & Basic Info ===
        'marketCapitalization': 'Market Capitalization',
        'beta': 'Beta',
        '52WeekHigh': '52-Week High',
        '52WeekLow': '52-Week Low',

        # === Valuation ===
        'peTTM': 'Price-to-Earnings (TTM)',
        'psTTM': 'Price-to-Sales (TTM)',
        'pbQuarterly': 'Price-to-Book (Quarterly)', # More recent than Annual
        'ptbvQuarterly': 'Price-to-Tangible Book Value (Quarterly)', # More recent
        'enterpriseValue': 'Enterprise Value', # Often used in EV ratios
        'currentEv/freeCashFlowTTM': 'EV/Free Cash Flow (TTM)', # Example EV ratio

        # === Profitability ===
        'grossMarginTTM': 'Gross Margin (TTM %)',
        'operatingMarginTTM': 'Operating Margin (TTM %)',
        'netProfitMarginTTM': 'Net Profit Margin (TTM %)',
        'roeTTM': 'Return on Equity (TTM %)',
        'roaTTM': 'Return on Assets (TTM %)',
        'roiTTM': 'Return on Investment (TTM %)',
        
        'roe5Y': 'Return on Equity (5Y Avg %)',
        'netProfitMargin5Y': 'Net Profit Margin (5Y Avg %)',

        # === Growth ===
        'revenueGrowthTTMYoy': 'Revenue Growth (TTM YoY %)',
        'revenueGrowthQuarterlyYoy': 'Revenue Growth (Quarterly YoY %)',
        'epsGrowthTTMYoy': 'EPS Growth (TTM YoY %)',
        'epsGrowthQuarterlyYoy': 'EPS Growth (Quarterly YoY %)',
        
        'revenueGrowth5Y': 'Revenue Growth (5Y %)',
        'epsGrowth5Y': 'EPS Growth (5Y %)',
        'dividendGrowthRate5Y': 'Dividend Growth Rate (5Y %)', # Also relevant to Dividends

        # === Financial Health / Solvency ===
        'currentRatioQuarterly': 'Current Ratio (Quarterly)', # More recent
        'quickRatioQuarterly': 'Quick Ratio (Quarterly)', # More recent
        'longTermDebt/equityQuarterly': 'Long-Term Debt/Equity (Quarterly)', # More recent
        'totalDebt/totalEquityQuarterly': 'Total Debt/Equity (Quarterly)', # More recent
        'netInterestCoverageTTM': 'Net Interest Coverage (TTM)', # Important for debt

        # === Dividends ===
        'dividendYieldIndicatedAnnual': 'Dividend Yield (Indicated Annual %)',
        'dividendPerShareTTM': 'Dividend Per Share (TTM)',
        'payoutRatioTTM': 'Payout Ratio (TTM %)',

        # === Per Share Data ===
        'epsTTM': 'Earnings Per Share (TTM)', # Key metric
        'bookValuePerShareQuarterly': 'Book Value Per Share (Quarterly)', # More recent
        'cashFlowPerShareTTM': 'Cash Flow Per Share (TTM)',
        'revenuePerShareTTM': 'Revenue Per Share (TTM)',

         # === Recent Stock Performance Context ===
        '52WeekPriceReturnDaily': '52-Week Price Return (%)',
        'yearToDatePriceReturnDaily': 'Year-to-Date Price Return (%)',
        '26WeekPriceReturnDaily': '26-Week Price Return (%)', # Medium term trend
        '13WeekPriceReturnDaily': '13-Week Price Return (%)', # Shorter medium term trend
        'priceRelativeToS&P50052Week': 'Price Relative to S&P500 (52 Week %)', # Market comparison
        'priceRelativeToS&P500Ytd': 'Price Relative to S&P500 (YTD %)', # Market comparison
    }

    formatted_string = ""
    
    for key, readable_name in metric_map.items():
        if key in data and data[key] is not None:
            value = data[key]
            
            formatted_value = value # Default
            if isinstance(value, float):
               
                if '%' in readable_name or \
                   'Margin' in readable_name or \
                   'Return' in readable_name or \
                   'Growth' in readable_name or \
                   'Yield' in readable_name or \
                   'Ratio' in readable_name or \
                   'Cagr' in readable_name or \
                   'CAGR' in readable_name:
                    
                    if 'Price-to-' not in readable_name and \
                       'Price/' not in readable_name and \
                       'EV/' not in readable_name and \
                       'Debt/Equity' not in readable_name and \
                       'Ratio' not in readable_name: # Keep ratios like Current Ratio as decimals
                         formatted_value = f"{value:.2f}%"
                    else:
                         formatted_value = f"{value:.2f}" # Format ratios/PE etc. to 2 decimal places
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
            
                 if 'Market Capitalization' in readable_name or 'Enterprise Value' in readable_name:
                      formatted_value = f"{value:,}" 
                 else:
                      formatted_value = f"{value}"

            formatted_string += f"**{readable_name}**: {formatted_value}\n"


    if not formatted_string:
        return f"No key financial data available to display for {ticker}."

    return formatted_string.strip()

def save_llm_interaction(prompt, response, news_data_available, ticker, base_filename='data/evaluation/query_and_prompt/llm_queries'):
    """
    Saves the LLM prompt and response to a CSV file, labeled by ticker and whether news data was available.

    Parameters:
    - prompt (str): The input prompt sent to the LLM.
    - response (str): The response received from the LLM.
    - news_data_available (bool): Flag indicating if news data was used.
    - ticker (str): Stock ticker symbol.
    - base_filename (str): Base name of the CSV file. Defaults to 'llm_queries'.
    """

    suffix = 'with_news' if news_data_available else 'without_news'
    filename = f'{base_filename}_{suffix}_updated.csv'

    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Ticker', 'Prompt', 'Response'])
        writer.writerow([ticker, prompt, response])

    print(f"Saved to {filename}")

def make_prediction(ticker, news_sentiment, labeled_news):
    """
    Generates a stock rating prediction using an LLM based on financial data and news sentiment.

    Args:
        ticker (str): The stock ticker symbol.
        news_sentiment (float or str): Overall sentiment score or label (e.g., "Positive").
        labeled_news (list or str): Pre-formatted list/string of recent labeled news items.
                                    Crucially, this should already be formatted for readability.

    Returns:
        str: The LLM's response text, hopefully containing the stock rating and analysis.
             Returns None if an error occurs.
    """
    # --- 1. Fetch and Validate Financial Data ---
    try:
        metrics = finnhub_client.get_company_basic_financials(ticker, 'all')
        # Validate Finnhub response
        if not metrics or 'metric' not in metrics or not isinstance(metrics.get('metric'), dict) or not metrics['metric']:
             st.error(f"Could not retrieve valid financial metric data for {ticker} from Finnhub.")
             print(f"Invalid Finnhub response for {ticker}: {metrics}") # Log the response
             return None
        financial_data_json = metrics['metric']
    except Exception as e:
        st.error(f"Error fetching financial data for {ticker} from Finnhub: {e}")
        return None

    # --- 2. Pre-process Data for LLM ---
    # Format financial data using the helper function
    readable_financial_data = format_financial_data(financial_data_json, ticker)
    if not readable_financial_data or "No key financial data available" in readable_financial_data :
         st.warning(f"Financial data formatting failed or returned no key metrics for {ticker}.")
         # Decide if you want to proceed without financials or return error
         # return None

    # Prepare News Data section (Requires YOUR specific formatting logic for news_sentiment and labeled_news)
    readable_labeled_news = ""
    overall_sentiment_summary = ""
    news_data_available = False

    if news_sentiment is not None and labeled_news: # Ensure labeled_news isn't empty/None
        news_data_available = True

        # Example: Format labeled_news if it's a list
        if isinstance(labeled_news, list):
             # Simple formatting, adjust as needed (e.g., handle dicts if news items are complex)
             readable_labeled_news = "\n".join([f"- {str(item).strip()}" for item in labeled_news])
        elif isinstance(labeled_news, str): # Assume already formatted if string
             readable_labeled_news = labeled_news
        else:
             readable_labeled_news = "Error: News data provided in unexpected format."

        # Example: Create a summary sentence for the overall sentiment
        if isinstance(news_sentiment, (int, float)):
            # Define thresholds for sentiment description
            if news_sentiment > 0.15: sentiment_desc = "Positive"
            elif news_sentiment < -0.15: sentiment_desc = "Negative"
            else: sentiment_desc = "Mixed/Neutral"
            overall_sentiment_summary = f"Overall sentiment from recent news is **{sentiment_desc}** (Average Score: {news_sentiment:.2f})."
        elif isinstance(news_sentiment, str): # If news_sentiment is already a label
             overall_sentiment_summary = f"Overall sentiment label from recent news: **{news_sentiment}**."
        else:
             overall_sentiment_summary = "Overall sentiment data provided in unexpected format."

    # --- 3. Construct the Prompt ---

    # 3a. Context Data Block: Only includes the prepared data.
    context_data = f"""**Context Data for {ticker}:**
                    **Key Financial Metrics:**
                    {readable_financial_data}
                    """
    if news_data_available:
        context_data += f"""
        **Recent News Summary:**
        {overall_sentiment_summary}
        Key Labeled News Items:
        {readable_labeled_news}
        """
    else:
        context_data += "\n**Recent News Summary:**\nNo recent news data available.\n"

    # 3b. Improved Instructions Block: Clearer, more detailed instructions.
    # Using {ticker} directly as this isn't nested inside another f-string here.
    instructions = f"""
        **Role:** Act as an objective financial analyst.

        **Objective:** Evaluate the investment merit of {ticker} based strictly on the context data provided and assign a corresponding stock rating (Strong Buy, Buy, Hold, Sell, or Strong Sell).

        **Task Instructions:**

        1.  **Determine Rating:** Assign a rating: Strong Buy, Buy, Hold, Sell, or Strong Sell.
        2.  **Synthesize and Analyze:**
            * **Assess Overall Merit:** Provide a concise assessment of the investment attractiveness of the company based *only* on the provided data.
            * **Identify Key Performance Indicators:** Pinpoint the critical positive and negative performance indicators evident in the financial metrics and news sentiment/items.
            * **Analyze Data Consistency:** Examine the alignment and discrepancies within the provided data (e.g., consistency between financial trends and recent news, areas of contradiction).
            * **Evaluate Opportunities and Challenges:** Based *solely* on the provided data, assess the significant potential opportunities and challenges facing the company.
        3.  **Justify Rating:**
            * **State Rating Rationale:** Clearly articulate the primary reasons supporting your assigned rating.
            * **Provide Data-Driven Evidence:** Substantiate your rationale by referencing specific **numerical values** and **bolded financial metric names**, as well as key themes or sentiment from the news items. Ensure you highlight the evidence supporting both positive and negative aspects where relevant.
            * **Explain Factor Prioritization:** Briefly explain the relative importance you assigned to different factors (e.g., the weight given to historical financial strength versus recent negative news, the significance of identified opportunities relative to existing risks) in reaching your conclusion.
        4.  **Format Output:**
            * Use Markdown.
            * Present the final rating *first*, using the exact format: `## {ticker}: :green[Strong Buy]`, `## {ticker}: :green[Buy]`, `## {ticker}: :orange[Hold]`, `## {ticker}: :red[Sell]`, or `## {ticker}: :red[Strong Sell]`.
            * Structure the subsequent analysis and justification under clear, descriptive Markdown headers (e.g., `### Investment Merit Assessment`, `### Key Performance Highlights`, `### Areas of Concern & Challenges`, `### Rating Justification`). Ensure logical flow.
            * Maintain an objective and analytical tone throughout.

        **Constraint Checklist:**
        * [ ] Rating based *only* on provided context?
        * [ ] External knowledge *not* used?
        * [ ] Specific data points referenced?
        * [ ] Final rating format correct?
        * [ ] Markdown headers used for structure?

        **Begin Response:**
        """

    # Combine context and instructions for the final prompt
    final_prompt = context_data + "\n---\n" + instructions # Added separator for clarity

    # --- 4. Send Prompt to LLM ---
    if 'chat_session' not in st.session_state:
        st.error("Chat session not initialized.")
        return None
    chat_session = st.session_state.chat_session

    # Debugging: Print the final prompt being sent
    # Consider using logging instead of print for production
    print(f"--- Sending Prompt to LLM for {ticker} ---")
    print(final_prompt)
    print("------------------------------------------")

    try:
        response = chat_session.send_message(final_prompt)
        # Basic check if response seems valid before returning
        if response and hasattr(response, 'text') and response.text:
             save_llm_interaction(final_prompt, response.text, news_data_available, ticker)
             return response.text
        else:
             st.warning(f"LLM returned an empty or invalid response for {ticker}.")
             print(f"Invalid LLM response object: {response}")
             return None
    except Exception as e:
        st.error(f"An error occurred while communicating with the LLM for {ticker}: {e}")
        return None

def make_comparative_rating(tickers, news_sentiments):
    metric_a = finnhub_client.get_company_basic_financials(tickers[0], 'all')
    metric_b = finnhub_client.get_company_basic_financials(tickers[1], 'all')
    financial_data_a_json = metric_a['metric']
    financial_data_b_json = metric_b['metric']

    if news_sentiments is not None:
        message = f"""Here are 2 companies; {tickers[0]} and {tickers[1]}.
        
                    The financial data for each companies are given here. 
                    {tickers[0]}: \n\n{financial_data_a_json}\n\n.
                    {tickers[1]}: \n\n{financial_data_b_json}\n\n.
                    The sentiment analysis of 20 of the most recent news surrounding the companies are {news_sentiments[0]} and {news_sentiments[1]} respectfully.
                    """
    else:
        message = f"""Here are 2 companies; {tickers[0]} and {tickers[1]}.
        
                    The financial data for each companies are given here. 
                    {tickers[0]}: \n\n{financial_data_a_json}\n\n.
                    {tickers[1]}: \n\n{financial_data_b_json}\n\n.
                    """
    chat_session = st.session_state.chat_session
    response = chat_session.send_message(f"""
**Role:** Objective financial analyst for comparative stock analysis.

**Objective:** Analyze financial data and news (if any) for {tickers} to provide comparative ratings (Buy, Hold, Sell for each) and identify the more favorable investment.

**Task Instructions:**

1. **Understand Message:** Analyze "{message}" containing financial data and news for {tickers}.

2. **Analyze Performance:** Compare key metrics and news for each ticker in {tickers}.

3. **Comparative Rating:** Assign a Buy, Hold, or Sell rating to each ticker.

4. **Synthesize (Comparative):**
   * **Overall Assessment:** Briefly compare investment merit.
   * **Key Indicators:** Highlight strengths and weaknesses for each based on data and news.
   * **Reasoning:** Explain ratings with **numerical values** and sentiment.
   * **Favorable Company:** Identify the better performer based on key metrics and sentiment.

5. **Justify:**
   * **Rating Rationale (Per Company):** State reasons for each rating.
   * **Data Evidence:** Reference **bolded, readable metrics** and sentiment from "{message}" if news is provided.
   * **Favorable Selection:** Explain why one company is more favorable, citing key differences.

6. **Format Output:**
   * Markdown.
   * Present the final rating *first*, using the exact format: `## {ticker}: :green[Strong Buy]`, `## {ticker}: :green[Buy]`, `## {ticker}: :orange[Hold]`, `## {ticker}: :red[Sell]`, or `## {ticker}: :red[Strong Sell]`.
   * Analysis under headers (e.g., `### Comparative Overview`, `### Strengths & Weaknesses`, `### Rating Justification`, `### Favorable Company`).
   * Objective tone.

**Constraint Checklist:**
* [ ] Comparative ratings for all tickers?
* [ ] Favorable company identified?
* [ ] External knowledge *not* used?
* [ ] Numerical data and sentiment referenced?
* [ ] Strengths/weaknesses for each?
* [ ] Key differentiators identified?
* [ ] Correct rating format?
* [ ] Markdown with headers?
""")

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
        help = """Tickers are short symbols (like AAPL for Apple or MSFT for Microsoft) used to identify companies on the stock market.
                This tool only supports companies in the S&P 500.""",
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
        st.page_link("tutorial.py", label="Tutorial")
        st.page_link("pages/2_Financial Ratio Analysis.py", label="Assisted Analysis")
        st.page_link("pages/4_News.py", label="News Analysis")
        st.page_link("pages/6_About.py", label="About")

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
        st.header('LLM Stock Rating')

        inner_col = st.columns(5, vertical_alignment="bottom")

        with inner_col[0]:
            rating_type_selectbox = st.selectbox(
                "Select rating method:",
                ['Single stock', 'Comparative rating'],
                index=0,
                key="rating_type_selectbox"
            )

        if rating_type_selectbox == 'Single stock':

            with inner_col[1]:
                predictive_selectbox = st.selectbox(
                    "Select a ticker to rate:",
                    st.session_state.selected_tickers,
                    index=0,
                    key="predictive_selectbox"
                )

            with inner_col[2]:
                include_news_selectbox = st.selectbox(
                    "Include news?:",
                    ['Yes', 'No'],
                    index=0,
                    key="include_news_selectbox"
                )

            with inner_col[3]:
                news_sentiment = aggregated_sentiment_df.loc[aggregated_sentiment_df['ticker'] == predictive_selectbox, 'average_sentiment'].values[0].capitalize()

                if st.button("Make prediction"):
                    with inner_col[4]:  # Place spinner in the next column
                        with st.spinner("Making prediction... Please wait", show_time=True):
                            if include_news_selectbox == "Yes":
                                labeled_news = get_labeled_news(predictive_selectbox)
                                # st.dataframe(labeled_news)
                                st.session_state.llm_analysis = make_prediction(predictive_selectbox, news_sentiment, labeled_news)
                            else:
                                st.session_state.llm_analysis = make_prediction(predictive_selectbox, None, None)

        elif rating_type_selectbox == 'Comparative rating':

            with inner_col[1]:
                predictive_multiselectbox = st.multiselect(
                    "Select ticker:",
                    st.session_state.selected_tickers,
                    key="predictive_multiselectbox",
                    max_selections=2
                )   

            with inner_col[2]:
                include_news_selectbox = st.selectbox(
                    "Include news?:",
                    ['Yes', 'No'],
                    index=0,
                    key="include_news_selectbox2"
                )

            with inner_col[3]:
                if st.button("Make prediction"):
                    if len(predictive_multiselectbox) == 0:
                        st.warning("Please select at least one ticker.")
                    else:
                        news_sentiments = []

                        for ticker in predictive_multiselectbox:
                            sentiment_value = aggregated_sentiment_df.loc[
                                aggregated_sentiment_df['ticker'] == ticker, 'average_sentiment'
                            ].values

                            if len(sentiment_value) > 0:
                                news_sentiments.append(sentiment_value[0].capitalize())
                            else:
                                news_sentiments.append("Neutral")  # Default if sentiment is missing
                        with inner_col[4]:  # Place spinner in the next column
                            with st.spinner("Making prediction... Please wait"):
                                if include_news_selectbox == "Yes":
                                    st.session_state.llm_analysis = make_comparative_rating(predictive_multiselectbox, news_sentiments)
                                else:
                                    st.session_state.llm_analysis = make_comparative_rating(predictive_multiselectbox, None)


        # st.dataframe(finaical_df)
        # st.write(news_sentiment)
    
        st.markdown(f"{st.session_state.llm_analysis}")

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
