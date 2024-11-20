import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title(":green[EasyStock] Learner :chart:")

data = pd.read_csv('Data/OID-Dataset.csv')
st.write(data)

st.write("Plotting using streamlit built in plotting library")

st.scatter_chart(
    data,
    x='Transaction Date',
    y="Price / share",
    color="Action",
)

st.write("Plotting using plotly")
fig = px.scatter(
    data,
    x="Transaction Date",
    y="Price / share",
    log_y=True,
    color="Action",
    hover_data=["Ticker", "Name", "No. of shares"],
    symbol="Action",
)

st.plotly_chart(fig, theme="streamlit", use_container_width=True)
