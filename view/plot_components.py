import pandas as pd
import streamlit as st
import plotly.express as px
from warnings import warn

# Might be deprecated

class Plot_Components:

    def draw_pe_ratio_component(self, api_client, api_config, API_KEY):
        warn('This method is deprecated.', DeprecationWarning, stacklevel=2)

        st.markdown("## Pick 2 tickers to compare P/E ratios")

        col1, col2 = st.columns(2)

        with col1:
            ticker1 = st.selectbox(
                "Select ticker:",
                api_config.get_ticker_options(),
                key="selectbox1"
            )

            st.markdown(f"You selected: :green[{ticker1}]")

            df1 = api_client.get_key_metrics(ticker1, API_KEY)

        with col2:
            ticker2 = st.selectbox(
                "Select ticker:",
                api_config.get_ticker_options(),
                key="selectbox2"
            )

            st.markdown(f"You selected: :green[{ticker2}]")

            df2 = api_client.get_key_metrics(ticker2, API_KEY)

        # Check if dataframes are not None before merging
        if df1 is not None and df2 is not None:
            merged_df = pd.concat([df1[['symbol', 'date', 'peRatio']], df2[['symbol', 'date', 'peRatio']]], ignore_index=True)
            merged_df['year'] = pd.to_datetime(merged_df['date']).dt.year
            fig = px.line(
                merged_df,
                x="year",
                y="peRatio",
                hover_data=['date', 'symbol', 'peRatio'],
                color="symbol"
            )
        else:
            st.error("Error: Could not load data for one or both tickers")
            fig = None

        st.plotly_chart(fig)
