import streamlit as st

# Page config
st.set_page_config(
    page_title="About",
    initial_sidebar_state="expanded"
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

############# PAGE STARTS HERE #############

with st.sidebar:
    st.title(":green[EasyStock] Learner :chart:")
    
    with st.container(border=True):
        st.header("Links to other pages")
        st.page_link("tutorial.py", label="Tutorial")
        st.page_link("pages/1_🏠_Homepage.py", label="Dashboard")
        st.page_link("pages/2_Financial Ratio Analysis.py", label="Assisted Analysis")
        st.page_link("pages/4_News.py", label="News Analysis")

st.title("About the Project")

st.markdown('''
    **EasyStock Learner** is an interactive tool designed to help users learn about investments in a fun and engaging way.
    The goal of the application is to simplify complex financial concepts and empower individuals to explore stock data and 
    analysis independently. By leveraging advanced technologies like artificial intelligence, this tool allows users to 
    gain valuable insights into stock performance and investment strategies. It offers an intuitive, beginner-friendly 
    experience while helping users develop a deeper understanding of how stock ratings, market trends, and financial metrics 
    influence investment decisions.

    This project aims to make learning about investments more accessible by providing clear, easy-to-understand explanations
    of stock ratings, sentiment analysis, and financial metrics. Whether you're a beginner or someone with a bit more knowledge,
    the tool provides an interactive platform for users to explore the financial world and gain insights based on real-time data.
''')

st.markdown('''
    This project was conceptualised as a dissertation project under the [University Of Nottingham](https://www.nottingham.ac.uk), focusing on the integration of 
    machine learning and financial data for educational purposes. As part of the dissertation, the tool was developed to 
    help both beginners and seasoned investors gain a better understanding of the financial markets and stock evaluations.

    The application integrates several external data sources and APIs to ensure that users receive the most up-to-date and 
    comprehensive stock data and news analysis:
    
    - **[Financial Modeling Prep (FMP)](https://site.financialmodelingprep.com)**
    - **[AlphaVantage](https://www.alphavantage.co)**
    - **[Finnhub](https://finnhub.io)**
    - **[Alpaca](https://docs.alpaca.markets/reference/authentication-2)**

    The decision to use multiple APIs was made to ensure the application provides a robust and reliable dataset, drawing 
    information from diverse sources. Each API serves a specific purpose, allowing users to access historical stock data, 
    real-time stock prices, financial statements, and news analysis. By combining data from these different providers, the 
    tool can offer a comprehensive view of the stock market, which helps users make more informed decisions.

    Additionally, the application incorporates **FinBERT**, a state-of-the-art machine learning model designed specifically 
    for **news sentiment analysis**. By analyzing the sentiment of the latest news articles, FinBERT allows the tool to 
    assess the overall sentiment surrounding each stock, giving users insights into how the market perceives current 
    events related to the selected stocks. Whether the sentiment is positive, neutral, or negative, users can easily 
    track the sentiment of stocks based on real-time news.

    For natural language processing and generating insights, this project relies on **Gemini-1.5-Flash**, a powerful language 
    model. Gemini-1.5-Flash is designed to process large amounts of data efficiently and provides the application with the 
    capability to generate meaningful stock predictions, sentiment analysis, and other customized insights. The choice of 
    Gemini-1.5-Flash was driven by its ability to process vast amounts of financial data and generate accurate, as well as being free to use.

    **EasyStock Learner** is built to help users understand the intricate world of stock trading and investing, making the 
    complex nature of financial markets more approachable and understandable for everyone.
''')
