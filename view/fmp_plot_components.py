import pandas as pd
import plotly.express as px
import streamlit as st

class FMP_Plot_Components:

    def draw_revenue(self, tickers, fmp_api_client_instance):
        """
        Draws a bar chart comparing revenue across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fmp_api_client_instance: FMP API client instance for data fetching

        Returns:
            None - Displays the plot using streamlit
        """
        revenue_dfs = []

        for ticker in tickers:
            incomeStatement_df = fmp_api_client_instance.get_income_statement(ticker)
            revenue_df = incomeStatement_df[['date', 'revenue']]
            revenue_df['ticker'] = ticker
            revenue_dfs.append(revenue_df)

        combined_revenue_df = pd.concat(revenue_dfs)

        fig = px.bar(
            combined_revenue_df,
            x="date",
            y="revenue",
            color='ticker',
            title="Revenue"
        )

        st.plotly_chart(fig)

    def draw_ebitda(self, tickers, fmp_api_client_instance):
        """
        Draws a bar chart comparing EBITDA across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fmp_api_client_instance: FMP API client instance for data fetching

        Returns:
            None - Displays the plot using streamlit
        """
        ebitda_dfs = []

        for ticker in tickers:
            incomeStatement_df = fmp_api_client_instance.get_income_statement(ticker)
            ebitda_df = incomeStatement_df[['date', 'ebitda']]
            ebitda_df['ticker'] = ticker
            ebitda_dfs.append(ebitda_df)

        combined_revenue_df = pd.concat(ebitda_dfs)

        fig = px.bar(
            combined_revenue_df,
            x="date",
            y="ebitda",
            color='ticker',
            title="EBITDA"
        )

        st.plotly_chart(fig)
