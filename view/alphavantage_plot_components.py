import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class AlphaVantage_Plot_Components:

    def draw_stock_prices(self, tickers, alphavantage_client):
        fig = go.Figure()

        for ticker in tickers:
            df = alphavantage_client.get_time_series_stock_prices(ticker)

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

                # Calculate and add median price line
                df['Median'] = (df['High'] + df['Low']) / 2
                fig.add_trace(go.Scatter(
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
            title="Daily Stock Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True,  # Show range slider
            template="plotly_dark",
            height=600
        )

        # Show plot (Streamlit or standalone)
        st.plotly_chart(fig, use_container_width = True)
