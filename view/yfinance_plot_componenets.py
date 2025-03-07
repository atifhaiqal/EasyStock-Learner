import pandas as pd
import plotly.express as px
import yfinance as yf
import streamlit as st

class YFinance_Plot_Components:

    def draw_debt(self, tickers):
        debt_dfs = []

        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                balancesheet_df = t.get_balance_sheet()
            except:
                continue

            # Convert index to 'date' column
            balancesheet_df = pd.DataFrame(balancesheet_df).reset_index()
            balancesheet_df = balancesheet_df.rename(columns={'index': 'date'})

            # Extract just the debt data
            debt_df = pd.DataFrame({
                'date': balancesheet_df['date'],
                'TotalDebt': balancesheet_df.loc[:,'TotalDebt'],
                'ticker': ticker
            })
            debt_dfs.append(debt_df)

        combined_debt_df = pd.concat(debt_dfs)

        fig = px.bar(
            combined_debt_df,
            x="date",
            y="TotalDebt",
            color='ticker',
            title="Total Debt"
        )

        st.plotly_chart(fig, use_container_width = True)
