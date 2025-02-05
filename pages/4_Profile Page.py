import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Profile Page",
    layout="wide",
)

# Initialising global variables
if "fmp_api_key" not in st.session_state:
    st.session_state["fmp_api_key"] = ""

if "av_api_key" not in st.session_state:
    st.session_state["av_api_key"] = ""

if "finnhub_api_key" not in st.session_state:
    st.session_state["finnhub_api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

############# PAGE STARTS HERE #############

st.title("Profile Page")

with st.container(border=True):
# Setup column for the profile
    col1, col2 = st.columns(2)

    with col1:
            # BugID: 1
        with st.popover("Edit profile"):
            user_name = st.text_input("Enter your username", st.session_state["user_name"])
            photo = st.file_uploader("Upload your profile picture (Optional):", type=["jpg","jpeg", "png"])
            if st.button("Submit"):
                st.session_state["user_name"] = user_name

        if photo:
            image = Image.open(photo)
            st.image(image, caption="Profile Picture", width=250)
        else:
            st.image("assets/default_pfp.jpg", caption="Profile Picture", width=250)

        if user_name:
            st.subheader(user_name)

    with col2:
        st.header("API Keys")

        # Note: the main page still uses hardcoding for API to make things easier to me when coding
        # CHANGE THIS LATER
        st.session_state["fmp_api_key"] = st.text_input("FMP API Key", type="password")
        st.session_state["av_api_key"] = st.text_input("AlphaVantage API Key", type="password")
        st.session_state["finnhub_api_key"] = st.text_input("FinnHub API Key", type="password")

st.header("API Keys Setup")

st.markdown('''
    This project uses a combination of [Financial Modeling Prep (FMP)](https://site.financialmodelingprep.com),
    [AlphaVantage](https://www.alphavantage.co), [Finnhub](https://finnhub.io) and
    [yfinance](https://ranaroussi.github.io/yfinance/index.html) APIs.
    In order to keep the project open-sourced, the application uses the free tier of all the APIs.
    Please follow the steps below for each API to create your own key and start learning.
''')

st.subheader("Financial Modeling Prep API")

st.markdown('''
    1. Visit the FMP Website
        - Go to [Financial Modeling Prep](https://site.financialmodelingprep.com).

    2. Sign Up or Log In
       	- If you don’t have an account, Sign Up.
    	- If you already have an account, Log In.

    3. Access API Page
        - After logging in, go to the API section from your account dashboard.

    4. Generate Your API Key
        - Click on “Get API Key” or “Generate Key”.
        - Your unique API key will be displayed.

    5. Copy the API Key
        - Copy your API key and paste it on the FMP API key input box in this page.
''')

st.subheader("Alpha Vantage API")

st.markdown('''
    1. Visit the Alpha Vantage Website
        - Go to [Alpha Vantage](https://www.alphavantage.co).

    2. Sign Up or Log In
        - If you don’t have an account, Sign Up by clicking the “Sign Up” button.
        - If you already have an account, Log In.

    3. Access API Key Section
        - After logging in, go to the API Keys section on your account page.

    4. Generate Your API Key
        - You will be provided with a unique API key. It should be displayed once you’re logged in.

    5. Copy the API Key
        - Copy the key and paste it on the Alpha Vantage API key input box in this page.
''')

st.subheader("Finnhub API")

st.markdown('''
    1. Visit the Finnhub Website
        - Go to [Finnhub](https://finnhub.io).

    2. Sign Up or Log In
        - If you don’t have an account, Sign Up by clicking the “Get started for free” button.
        - If you already have an account, Log In.

    3. Access API Key Section
        - Once logged in, go to your Dashboard.
        - In the dashboard, find the section labeled “API Key”.

    4. Generate Your API Key
        - Finnhub will provide you with a unique API key. It will be visible in the “API Key” section.

    5. Copy the API Key
        - Copy the key and paste it on the Finnhub API key input box in this page.

''')
