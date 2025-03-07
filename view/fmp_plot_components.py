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
            try:
                incomeStatement_df = fmp_api_client_instance.get_income_statement(ticker)
            except:
                continue
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

        st.plotly_chart(fig, use_container_width = True)

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
            try:
                incomeStatement_df = fmp_api_client_instance.get_income_statement(ticker)
            except:
                continue
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

        st.plotly_chart(fig, use_container_width = True)

    def draw_total_debt(self, tickers, fmp_api_client_instance):
        """
        Draws a bar chart comparing EBITDA across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fmp_api_client_instance: FMP API client instance for data fetching

        Returns:
            None - Displays the plot using streamlit
        """
        totaldebt_dfs = []

        for ticker in tickers:
            try:
                balancesheet_df = fmp_api_client_instance.get_balance_sheet(ticker)
            except:
                continue
            totaldebt_df = balancesheet_df[['date', 'totalDebt']]
            totaldebt_df['ticker'] = ticker
            totaldebt_dfs.append(totaldebt_df)

        combined_totaldebt_df = pd.concat(totaldebt_dfs)

        fig = px.bar(
            combined_totaldebt_df,
            x="date",
            y="totalDebt",
            color='ticker',
            title="Total Debt"
        )

        st.plotly_chart(fig, use_container_width = True)

    def draw_net_debt(self, tickers, fmp_api_client_instance):
        """
        Draws a bar chart comparing EBITDA across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fmp_api_client_instance: FMP API client instance for data fetching

        Returns:
            None - Displays the plot using streamlit
        """
        netdebt_dfs = []

        for ticker in tickers:
            try:
                balancesheet_df = fmp_api_client_instance.get_balance_sheet(ticker)
            except:
                continue
            netdebt_df = balancesheet_df[['date', 'netDebt']]
            netdebt_df['ticker'] = ticker
            netdebt_dfs.append(netdebt_df)

        combined_totaldebt_df = pd.concat(netdebt_dfs)

        fig = px.bar(
            combined_totaldebt_df,
            x="date",
            y="netDebt",
            color='ticker',
            title="Net Debt"
        )

        st.plotly_chart(fig, use_container_width = True)

    def draw_longterm_debt(self, tickers, fmp_api_client_instance):
        """
        Draws a bar chart comparing EBITDA across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fmp_api_client_instance: FMP API client instance for data fetching

        Returns:
            None - Displays the plot using streamlit
        """
        longtermdebt_dfs = []

        for ticker in tickers:
            try:
                balancesheet_df = fmp_api_client_instance.get_balance_sheet(ticker)
            except:
                continue
            longtermdebt_df = balancesheet_df[['date', 'longTermDebt']]
            longtermdebt_df['ticker'] = ticker
            longtermdebt_dfs.append(longtermdebt_df)

        combined_totaldebt_df = pd.concat(longtermdebt_dfs)

        fig = px.bar(
            combined_totaldebt_df,
            x="date",
            y="longTermDebt",
            color='ticker',
            title="Long Term Debt"
        )

        st.plotly_chart(fig, use_container_width = True)

    def draw_net_income(self, tickers, fmp_api_client_instance):
        """
        Draws a bar chart comparing EBITDA across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fmp_api_client_instance: FMP API client instance for data fetching

        Returns:
            None - Displays the plot using streamlit
        """
        netincome_dfs = []

        for ticker in tickers:
            try:
                balancesheet_df = fmp_api_client_instance.get_income_statement(ticker)
            except:
                continue
            netincome_df = balancesheet_df[['date', 'netIncome']]
            netincome_df['ticker'] = ticker
            netincome_dfs.append(netincome_df)

        combined_netincome_df = pd.concat(netincome_dfs)

        fig = px.bar(
            combined_netincome_df,
            x="date",
            y="netIncome",
            color='ticker',
            title="Net Income"
        )

        st.plotly_chart(fig, use_container_width = True)

    def draw_net_income_ratio(self, tickers, fmp_api_client_instance):
        """
        Draws a bar chart comparing EBITDA across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fmp_api_client_instance: FMP API client instance for data fetching

        Returns:
            None - Displays the plot using streamlit
        """
        netincomeratio_dfs = []

        for ticker in tickers:
            try:
                balancesheet_df = fmp_api_client_instance.get_income_statement(ticker)
            except:
                continue
            netincomeratio_df = balancesheet_df[['date', 'netIncomeRatio']]
            netincomeratio_df['ticker'] = ticker
            netincomeratio_dfs.append(netincomeratio_df)

        combined_netincome_df = pd.concat(netincomeratio_dfs)

        fig = px.bar(
            combined_netincome_df,
            x="date",
            y="netIncomeRatio",
            color='ticker',
            title="Net Income Ratio"
        )

        st.plotly_chart(fig, use_container_width = True)

    def draw_net_change_in_cash(self, tickers, fmp_api_client_instance):
        """
        Draws a bar chart comparing EBITDA across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fmp_api_client_instance: FMP API client instance for data fetching

        Returns:
            None - Displays the plot using streamlit
        """
        netchangeincash_dfs = []

        for ticker in tickers:
            try:
                cashflow_df = fmp_api_client_instance.get_cashflow_statement(ticker)
            except:
                continue
            netchangeincash_df = cashflow_df[['date', 'netChangeInCash']]
            netchangeincash_df['ticker'] = ticker
            netchangeincash_dfs.append(netchangeincash_df)

        combined_netchangeincash_df = pd.concat(netchangeincash_dfs)

        fig = px.bar(
            combined_netchangeincash_df,
            x="date",
            y="netChangeInCash",
            color='ticker',
            title="Net Change In Cash"
        )

        st.plotly_chart(fig, use_container_width = True)
