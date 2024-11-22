import streamlit as st

st.set_page_config(
    page_title="Profile Page",
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

############# PAGE STARTS HERE #############

st.title("Profile Page")

# Setup column for the page
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        # BugID: 1
        with st.popover("Edit profile"):
            user_name = st.text_input("Your Name", st.session_state["user_name"])
            api_key = st.text_input("API Key", st.session_state["api_key"])
            if st.button("Submit"):
                st.session_state["user_name"] = user_name
                st.session_state["api_key"] = api_key

        st.write("#### Nickname: ", user_name)
        st.write("##### API Key: ", api_key)

with col2:
    st.header("Financial Modeling Prep API")
    st.markdown('''
        This project uses the [Financial Modeling Prep (FMP)](https://site.financialmodelingprep.com) API.
        FMP is a company that provides stock market data as well as free and paid financial APIs.
        In order to keep the project open-sourced, the application uses the free tier of FMP's API key.
        Please follow the steps below to create your own key and start learning.
    ''')

    st.header("API Key Setup")
    st.markdown('''
        1. <to do: write steps>
    ''')
