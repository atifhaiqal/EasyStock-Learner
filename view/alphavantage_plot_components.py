import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


class AlphaVantage_Plot_Components:

    def draw_stock_prices(self, tickers, show_candlestick ,alphavantage_client):
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

                if show_candlestick:
                    # Add candlestick trace for this ticker
                    fig.add_trace(go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],

                        name=ticker
                    ))

                # Calculate and add median price line
                df['Median'] = (df['High'] + df['Low']) / 2
                fig.add_trace(go.Line(
                    x=df["Date"],
                    y=df["Median"],
                    mode='lines',
                    name=f'{ticker} Median',
                    line=dict(width=1),
                    opacity=0.6
                ))
            else:
                st.write(f"Failed to fetch stock data for {ticker}")

        # Customize layout
        fig.update_layout(
            title="Daily Stock Prices (Weekdays only)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,  # Show range slider
            template="plotly_dark",
            height=400,
            margin=dict(l=5, r=5, t=25, b=0)
            # xaxis_rangebreaks=[dict(bounds=["sat", "mon"])]  # Hide weekends
        )

        # Show plot
        st.plotly_chart(fig, use_container_width = True)

    def draw_volume_with_moving_average(self, tickers, alphavantage_client):
        """
        Draw an interactive chart visualizing the stock volume data with a 5-day moving average.

        This function plots the volume of stocks traded along with the 5-day moving average of the volume
        for each ticker using Plotly, with distinct colors for each stock.

        Args:
            tickers (list): List of stock ticker symbols to plot (e.g. ['AAPL', 'MSFT'])
            alphavantage_client (AlphaVantageClient): Instance of AlphaVantage API client used to fetch stock data

        Returns:
            None: Renders an interactive Plotly chart in the Streamlit app
        """
        # Define a color palette for different stocks
        colors = [
            "#FF6F61", "#FFD700", "#4CAF50", "#1E90FF", "#8A2BE2", 
            "#FF1493", "#20B2AA", "#FF4500", "#2E8B57", "#9370DB"
        ]

        fig = go.Figure()

        for i, ticker in enumerate(tickers):
            try:
                df = alphavantage_client.get_time_series_stock_prices(ticker)
            except:
                continue

            if df is not None:
                # Reset index for plotting
                df.reset_index(inplace=True)
                df.rename(columns={"index": "Date"}, inplace=True)

                # Assign colors based on index
                volume_color = colors[i % len(colors)]  
                ma_color = colors[(i) % len(colors)] 

                # Plot the volume as a bar chart
                fig.add_trace(go.Bar(
                    x=df["Date"],
                    y=df["Volume"],
                    name=f'{ticker} Volume',
                    marker=dict(color=volume_color, opacity=0.5)
                ))

                # Calculate 5-day moving average for volume
                df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()

                # Plot the 5-day moving average of the volume as a line
                fig.add_trace(go.Scatter(
                    x=df["Date"],
                    y=df["Volume_MA5"],
                    mode='lines',
                    name=f'{ticker} 5-Day Volume MA',
                    line=dict(width=2, color=ma_color),
                    opacity=0.9
                ))

            else:
                st.write(f"Failed to fetch stock data for {ticker}")

        # Customize layout for the volume chart
        fig.update_layout(
            title="Stock Volume with 5-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Volume",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=200,  # Adjust height for clarity
            showlegend=True,
            margin=dict(l=6, r=6, t=25, b=0)
        )

        # Show the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def draw_combined_price_volume_chart(self, tickers, show_candlestick, show_volume, alphavantage_client):
        """
        Draws a combined interactive chart with stock prices (candlesticks) and volume bars
        using separate y-axes. Price axis on left, volume axis on right.
        
        Args:
            tickers (list): Stock tickers to plot (e.g. ['AAPL', 'MSFT'])
            alphavantage_client (AlphaVantageClient): API client for stock data
        """
        

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Color palette for volume bars (different from price colors)
        volume_colors = [
            "#FF6F61", "#FFD700", "#4CAF50", "#1E90FF", 
            "#8A2BE2", "#FF1493", "#20B2AA", "#FF4500"
        ]

        for idx, ticker in enumerate(tickers):
            try:
                df = alphavantage_client.get_time_series_stock_prices(ticker)
                if df is None:
                    continue
                    
                df = df.reset_index().rename(columns={"index": "Date"})

                # Add Price Trace (Candlestick)
                if show_candlestick:
                    fig.add_trace(go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name=f'{ticker} Price',
                        increasing_line_color='#2ECC71',  # Green
                        decreasing_line_color='#E74C3C'   # Red
                    ), secondary_y=False)

                # Calculate and add median price line
                df['Median'] = (df['High'] + df['Low']) / 2
                fig.add_trace(go.Line(
                    x=df["Date"],
                    y=df["Median"],
                    mode='lines',
                    name=f'{ticker} Median',
                    line=dict(width=1),
                    opacity=0.6
                ))

                # Add Volume Trace (Bars)
                if show_volume:
                    fig.add_trace(go.Bar(
                        x=df["Date"],
                        y=df["Volume"],
                        name=f'{ticker} Volume',
                        marker_color=volume_colors[idx % len(volume_colors)],
                        opacity=0.5,
                        showlegend=False  # Reduce legend clutter
                    ), secondary_y=True)

            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")

        # Layout Configuration
        fig.update_layout(
            title="Price and Volume Analysis",
            template="plotly_dark",
            height=600,
            hovermode="x unified",
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=6, r=6, t=6, b=6)
        )

        # Axis Settings
        fig.update_yaxes(
            title_text="Price (USD)",
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        )

        fig.update_yaxes(
            title_text="Volume",
            secondary_y=True,
            showgrid=False,
            overlaying="y",
            side="right"
        )

        st.plotly_chart(fig, use_container_width=True)