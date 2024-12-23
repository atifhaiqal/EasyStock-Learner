import pandas as pd
import plotly.express as px
import streamlit as st

class Finnhub_Plot_Components:

    def draw_pe_ratio(self, tickers, finnhub_client):
        """
        Draw a line plot of PE ratios over time for multiple stock tickers.

        Args:
            tickers (list): List of stock ticker symbols to plot
            finnhub_client: Finnhub API client instance for fetching financial data

        Returns:
            None - Displays the plot using streamlit
        """
        # Create empty dataframe to store all PE ratios
        combined_df = pd.DataFrame()

        # Fetch PE ratio data for each ticker
        for ticker in tickers:
            metrics = finnhub_client.company_basic_financials(ticker, 'all')
            peRatio = metrics["series"]["annual"]["pe"]
            df = pd.json_normalize(peRatio)
            df['ticker'] = ticker # Add column to identify ticker
            combined_df = pd.concat([combined_df, df])

        fig = px.line(
            combined_df,
            x="period",
            y="v",
            color='ticker',
            title="PE RATIO"
        )

        st.plotly_chart(fig)
