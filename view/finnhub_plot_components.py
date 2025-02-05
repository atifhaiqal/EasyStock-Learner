import pandas as pd
import plotly.express as px
import streamlit as st

class Finnhub_Plot_Components:

    def draw_pe_ratio(self, tickers, finnhub_client):
        """
        Draws a line plot showing PE (Price to Earnings) ratios over time for multiple stocks.

        This method fetches PE ratio data from Finnhub for each provided ticker symbol and creates
        a line plot comparing the PE ratios over time. If data cannot be retrieved for a ticker,
        it will show as 0.

        Args:
            tickers (list): List of stock ticker symbols (e.g. ['AAPL', 'MSFT'])
            finnhub_client (finnhub.Client): Initialized Finnhub API client object

        Returns:
            None

        Displays:
            A Plotly line chart via Streamlit showing PE ratios over time for each ticker
        """
        # Create empty dataframe to store all PE ratios
        combined_df = pd.DataFrame()

        # Fetch PE ratio data for each ticker
        for ticker in tickers:
            try:
                metrics = finnhub_client.company_basic_financials(ticker, 'all')
                peRatio = metrics["series"]["annual"]["pe"]
                df = pd.json_normalize(peRatio)
                df['ticker'] = ticker # Add column to identify ticker
                combined_df = pd.concat([combined_df, df])
            except (KeyError, TypeError):
                df = pd.DataFrame({'period': [None], 'v': [0], 'ticker': [ticker]})
                combined_df = pd.concat([combined_df, df])

        fig = px.line(
            combined_df,
            x="period",
            y="v",
            color='ticker',
            title="PE RATIO"
        )

        st.plotly_chart(fig, use_container_width = True)

    def draw_pb_ratio(self, tickers, finnhub_client):
        """
        Draws a line plot showing PB (Price to Book) ratios over time for multiple stocks.

        This method fetches PB ratio data from Finnhub for each provided ticker symbol and creates
        a line plot comparing the PB ratios over time. If data cannot be retrieved for a ticker,
        it will show as 0.

        Args:
            tickers (list): List of stock ticker symbols (e.g. ['AAPL', 'MSFT'])
            finnhub_client (finnhub.Client): Initialized Finnhub API client object

        Returns:
            None

        Displays:
            A Plotly line chart via Streamlit showing PB ratios over time for each ticker
        """
        # Create empty dataframe to store all PE ratios
        combined_df = pd.DataFrame()

        # Fetch PE ratio data for each ticker
        for ticker in tickers:
            try:
                metrics = finnhub_client.company_basic_financials(ticker, 'all')
                pbRatio = metrics["series"]["annual"]["pb"]
                df = pd.json_normalize(pbRatio)
                df['ticker'] = ticker # Add column to identify ticker
                combined_df = pd.concat([combined_df, df])
            except (KeyError, TypeError):
                df = pd.DataFrame({'period': [None], 'v': [0], 'ticker': [ticker]})
                combined_df = pd.concat([combined_df, df])

        fig = px.line(
            combined_df,
            x="period",
            y="v",
            color='ticker',
            title="PB RATIO"
        )

        st.plotly_chart(fig, use_container_width = True)

    def draw_eps_ratio(self, tickers, finnhub_client):
        """
        Draws a line plot showing EPS (Earnings Per Share) over time for multiple stocks.

        This method fetches EPS data from Finnhub for each provided ticker symbol and creates
        a line plot comparing the EPS values over time. If data cannot be retrieved for a ticker,
        it will show as 0.

        Args:
            tickers (list): List of stock ticker symbols (e.g. ['AAPL', 'MSFT'])
            finnhub_client (finnhub.Client): Initialized Finnhub API client object

        Returns:
            None

        Displays:
            A Plotly line chart via Streamlit showing EPS over time for each ticker
        """
        # Create empty dataframe to store all PE ratios
        combined_df = pd.DataFrame()

        # Fetch PE ratio data for each ticker
        for ticker in tickers:
            try:
                metrics = finnhub_client.company_basic_financials(ticker, 'all')
                epsRatio = metrics["series"]["annual"]["eps"]
                df = pd.json_normalize(epsRatio)
                df['ticker'] = ticker # Add column to identify ticker
                combined_df = pd.concat([combined_df, df])
            except (KeyError, TypeError):
                df = pd.DataFrame({'period': [None], 'v': [0], 'ticker': [ticker]})
                combined_df = pd.concat([combined_df, df])

        fig = px.line(
            combined_df,
            x="period",
            y="v",
            color='ticker',
            title="EPS RATIO"
        )

        st.plotly_chart(fig, use_container_width = True)

    def draw_dividend_yield_annual(self, tickers, finnhub_client):
        """
        Draws a bar plot showing annual dividend yields for multiple stocks.

        This method fetches the indicated annual dividend yield from Finnhub for each provided
        ticker symbol and creates a bar plot comparing the yields. If data cannot be retrieved
        for a ticker, it will show as 0.

        Args:
            tickers (list): List of stock ticker symbols (e.g. ['AAPL', 'MSFT'])
            finnhub_client (finnhub.Client): Initialized Finnhub API client object

        Returns:
            None

        Displays:
            A Plotly bar chart via Streamlit showing dividend yields for each ticker
        """
        # Create empty dataframe to store all PE ratios
        combined_df = pd.DataFrame()

        # Fetch PE ratio data for each ticker
        for ticker in tickers:
            try:
                metrics = finnhub_client.company_basic_financials(ticker, 'all')
                dividendYield = metrics["metric"]["dividendYieldIndicatedAnnual"]
                df = pd.DataFrame({'dividendYieldIndicatedAnnual': [dividendYield], 'ticker': [ticker]})
                combined_df = pd.concat([combined_df, df])
            except (KeyError, TypeError):
                df = pd.DataFrame({'dividendYieldIndicatedAnnual': [0], 'ticker': [ticker]})
                combined_df = pd.concat([combined_df, df])

        fig = px.bar(
            combined_df,
            x="ticker",
            y="dividendYieldIndicatedAnnual",
            color='ticker',
            title="Dividend Yield Indicated Annual"
        )

        st.plotly_chart(fig, use_container_width = True)
