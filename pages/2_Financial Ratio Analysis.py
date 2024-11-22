import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Financial Ratio Analysis",
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

############# PAGE STARTS HERE #############

st.title("Financial Ratio Analysis")
