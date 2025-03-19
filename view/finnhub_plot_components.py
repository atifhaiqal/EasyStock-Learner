import pandas as pd
import plotly.express as px
import streamlit as st
import altair as alt

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
                metrics = finnhub_client.get_company_basic_financials(ticker, 'all')
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

        return combined_df

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
                metrics = finnhub_client.get_company_basic_financials(ticker, 'all')
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

        return combined_df

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
                metrics = finnhub_client.get_company_basic_financials(ticker, 'all')
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

        return combined_df

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
                metrics = finnhub_client.get_company_basic_financials(ticker, 'all')
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

        return combined_df

    def draw_stock_ratings(self, tickers, finnhub_client):
        """
        Draws an interactive grouped horizontal bar chart showing stock ratings
        (Strong Buy, Buy, Hold, Sell, Strong Sell) for multiple stocks.

        Users can click on a rating type to adjust the graph so that the selected rating starts at 0.

        Args:
            tickers (list): List of stock ticker symbols (e.g. ['AAPL', 'MSFT'])
            finnhub_client (finnhub.Client): Initialized Finnhub API client object
        """

        # Define the fixed order for rating types
        rating_order = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]

        # Create empty DataFrame to store ratings
        combined_df = pd.DataFrame()

        # Fetch stock ratings for each ticker
        for ticker in tickers:
            try:
                ratings = finnhub_client.get_recommendation_trends(ticker)
                if ratings:
                    latest_rating = ratings[-1]  # Get the most recent rating data
                    df = pd.DataFrame({
                        'Ticker': [ticker] * 5,
                        'Rating Type': rating_order,
                        'Count': [
                            latest_rating.get("strongBuy", 0),
                            latest_rating.get("buy", 0),
                            latest_rating.get("hold", 0),
                            latest_rating.get("sell", 0),
                            latest_rating.get("strongSell", 0)
                        ]
                    })
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
            except (KeyError, TypeError, IndexError):
                pass

        if combined_df.empty:
            st.warning("No stock rating data available.")
            return None

        # Custom color mapping
        rating_colors_dict = {
            "Strong Buy": "#067E00",  # Dark Green
            "Buy": "#3BD133",  # Green
            "Hold": "#FBF909",  # Yellow
            "Sell": "#FBAA09",  # Pink
            "Strong Sell": "#FB1509"  # Red
        }

        # Get colors in correct order matching rating categories
        plotly_colors = [rating_colors_dict[rating] for rating in rating_order]

        # Create grouped horizontal bar chart
        fig = px.bar(
            combined_df,
            x="Count",
            y="Ticker",
            color="Rating Type",
            barmode="group",
            color_discrete_sequence=plotly_colors,
            title="Stock Ratings by Ticker",
            category_orders={"Rating Type": rating_order},  # Enforce order
            height=400
        )

        fig.update_layout(xaxis_title="Number of Ratings", yaxis_title="Stock Ticker")

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        return combined_df

    def draw_consensus_ratings(self, tickers, finnhub_client):
        st.markdown("### Consensus Ratings")

        for ticker in tickers:
            try:
                rating = finnhub_client.get_rating_consensus(ticker)

                if(rating == 'Buy'):
                    st.markdown(f"#### {ticker}: :green[{rating}] ")
                elif(rating == 'Hold'):
                    st.markdown(f"#### {ticker}: :orange[{rating}] ")
                elif(rating == 'Sell'):
                    st.markdown(f"#### {ticker}: :red[{rating}] ")

            except (KeyError, TypeError):
                st.write(f"{ticker}: :red[NONE] ")
