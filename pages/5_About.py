import streamlit as st

# Page config
st.set_page_config(
    page_title="About",
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

############# PAGE STARTS HERE #############

st.title("About the project")

st.markdown('''
    EasyStock Learner is an interactive tool designed to help people learn about investments.
    It offers a sandbox experience, allowing users to explore financial data independently.
    Leveraging the power of artificial intelligence, the tool provides clear explanations
    for stock ratings and highlights key data points, helping users understand the reasoning
    behind investment decisions while gaining valuable insights.
''')

st.markdown('''
    This project uses a combination of [Financial Modeling Prep (FMP)](https://site.financialmodelingprep.com),
    [AlphaVantage](https://www.alphavantage.co), [Finnhub](https://finnhub.io) and
    [yfinance](https://ranaroussi.github.io/yfinance/index.html) APIs.
    In order to keep the project open-sourced, the application uses the free tier of all the APIs.
    Please follow the steps in the profile page to create your own key before using the application.
''')

st.markdown('''
    This project was conceptualised as a dissertation project under the [University Of Nottingham](https://www.nottingham.ac.uk).
''')
