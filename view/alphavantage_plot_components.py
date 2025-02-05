import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


class AlphaVantage_Plot_Components:

    def draw_stock_prices(self, tickers, alphavantage_client):
        """
        Draw an interactive candlestick chart visualizing stock price data.

        This function creates and displays an interactive candlestick chart showing stock
        price data for one or more tickers using Plotly. For each ticker, it plots both
        candlestick price data and a median price line.

        Args:
            tickers (list): List of stock ticker symbols to plot (e.g. ['AAPL', 'MSFT'])
            alphavantage_client (AlphaVantageClient): Instance of AlphaVantage API client
                used to fetch stock data

        Returns:
            None: Renders an interactive Plotly chart in the Streamlit app

        Examples:
            >>> plotter = AlphaVantage_Plot_Components()
            >>> plotter.draw_stock_prices(['AAPL', 'MSFT'], av_client)

        The rendered chart includes:
            - Candlestick chart showing OHLC prices for each ticker
            - Median price line for each ticker
            - Interactive date range slider
            - Hover tooltips with price details
            - Dark theme styling
            - Automatic responsiveness to container width
        """
        fig = go.Figure()

        for ticker in tickers:
            try:
                df = alphavantage_client.get_time_series_stock_prices(ticker)
            except:
                continue

            if df is not None:
                # Reset index for plotting
                df.reset_index(inplace=True)
                df.rename(columns={"index": "Date"}, inplace=True)

                # Add candlestick trace for this ticker
                fig.add_trace(go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name=ticker
                ))

                # # Calculate and add median price line
                # df['Median'] = (df['High'] + df['Low']) / 2
                # fig.add_trace(go.Scatter(
                #     x=df["Date"],
                #     y=df["Median"],
                #     mode='lines',
                #     name=f'{ticker} Median',
                #     line=dict(width=1),
                #     opacity=0.6
                # ))
            else:
                st.write(f"Failed to fetch stock data for {ticker}")

        # Customize layout
        fig.update_layout(
            title="Daily Stock Prices (Weekdays only)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True,  # Show range slider
            template="plotly_dark",
            height=600,
            # xaxis_rangebreaks=[dict(bounds=["sat", "mon"])]  # Hide weekends
        )

        # Show plot (Streamlit or standalone)
        st.plotly_chart(fig, use_container_width = True)
