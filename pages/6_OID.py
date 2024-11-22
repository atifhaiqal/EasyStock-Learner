import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Open Investment Bank Data")

data = pd.read_csv('Data/OID-Dataset.csv')
st.write(data)

st.subheader("Plotting using streamlit built in plotting library")

st.scatter_chart(
    data,
    x='Transaction Date',
    y="Price / share",
    color="Action",
)

st.subheader("Plotting using plotly")

is_Log = st.checkbox("Use log scale for y axis")

fig = px.scatter(
    data,
    x="Transaction Date",
    y="Price / share",
    log_y=is_Log,
    color="Action",
    hover_data=["Ticker", "Name", "No. of shares"],
    symbol="Action",
)

st.plotly_chart(fig, theme="streamlit", use_container_width=True)
