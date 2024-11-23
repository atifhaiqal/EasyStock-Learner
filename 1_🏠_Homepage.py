import streamlit as st
from config.settings import options
import numpy as np
import pandas as pd
import plotly.express as px


# Page config
st.set_page_config(
    page_title="EasyStock Learner",
    page_icon= "💹",
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

############# PAGE STARTS HERE #############

st.title(":green[EasyStock] Learner :chart:")

if st.session_state["user_name"] and "" or st.session_state["api_key"] != "":
    st.write("## Hi ", st.session_state["user_name"], "!")
else:
    st.write("## Please proceed to the profile page to set up API key and nickname")

st.markdown("## Pick stocks to compare")

selected_stocks = st.multiselect("Select stocks:", options, max_selections=3)
