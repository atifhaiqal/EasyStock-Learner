from services.finnhub_api_client import get_finnhub_client
import streamlit as st
from config.api_settings import FMP_APIConfig
import finnhub
import google.generativeai as genai
import pandas as pd
import plotly.express as px

# importing api clients
from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient, get_fmp_client
from services.alphavantage_api_client import AlphaVantage_APIClient, get_alphavantage_client
from config.api_settings import APIConfig

# importing plot components
from view.alphavantage_plot_components import AlphaVantage_Plot_Components
from view.finnhub_plot_components import Finnhub_Plot_Components
from view.fmp_plot_components import FMP_Plot_Components


############# PAGE CONFIG #############
st.set_page_config(
    page_title="EasyStock Learner",
    page_icon= "💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ['AAPL']

genai.configure(api_key="AIzaSyDeEKkLTe_Gbv0jTn4Ormx5OUy8cuz8ahA")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="You are a an assistant in a investment learning app called EasyStock Learner. Your role is to predict the stock rating of a company given its financial data and give the reasoning behind the rating (either buy or sell). Apart from that your role is also to help users understand the meaning behind the different metrics and ratios in finance. You are required to analyse time series financial data and generate an explanation of the performance of the metric. Be concise and start directly with insights. Explain in a tone thats easy to understand for someone with little to moderate financial litteracy.",
)

if "chat_session" not in st.session_state:
    st.session_state.chat_session = genai.ChatSession(model)

if "llm_explaination" not in st.session_state:
    st.session_state.llm_explaination = "No explanations currently."

#temp
FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'
FINNHUB_API_KEY = 'ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg'

fmp_api = get_fmp_client(FMP_API_KEY)
av_api = get_alphavantage_client(AV_API_KEY)
finnhub_client = get_finnhub_client(FINNHUB_API_KEY)
api_config = FMP_APIConfig()

fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()
av_plot = AlphaVantage_Plot_Components()

income_statement_list = {
    'revenue': 'Revenue',
    'costOfRevenue': 'Cost of Revenue',
    'grossProfit': 'Gross Profit',
    'grossProfitRatio': 'Gross Profit Ratio',
    'researchAndDevelopmentExpenses': 'R&D Expenses',
    'generalAndAdministrativeExpenses': 'G&A Expenses',
    'sellingAndMarketingExpenses': 'Sales & Marketing Expenses',
    'sellingGeneralAndAdministrativeExpenses': 'SG&A Expenses',
    'otherExpenses': 'Other Expenses',
    'operatingExpenses': 'Operating Expenses',
    'costAndExpenses': 'Total Costs & Expenses',
    'interestIncome': 'Interest Income',
    'interestExpense': 'Interest Expense',
    'depreciationAndAmortization': 'Depreciation & Amortization',
    'ebitda': 'EBITDA',
    'ebitdaratio': 'EBITDA Ratio',
    'operatingIncome': 'Operating Income',
    'operatingIncomeRatio': 'Operating Income Ratio',
    'totalOtherIncomeExpensesNet': 'Total Other Income/Expenses (Net)',
    'incomeBeforeTax': 'Income Before Tax',
    'incomeBeforeTaxRatio': 'Income Before Tax Ratio',
    'incomeTaxExpense': 'Income Tax Expense',
    'netIncome': 'Net Income',
    'netIncomeRatio': 'Net Income Ratio',
    'eps': 'EPS (Basic)',
    'epsdiluted': 'EPS (Diluted)',
    'weightedAverageShsOut': 'Weighted Avg Shares Out',
    'weightedAverageShsOutDil': 'Weighted Avg Shares Out (Diluted)'
}

balance_sheet_list = {
    'cashAndCashEquivalents': 'Cash & Cash Equivalents',
    'shortTermInvestments': 'Short-Term Investments',
    'cashAndShortTermInvestments': 'Cash & Short-Term Investments',
    'netReceivables': 'Net Receivables',
    'inventory': 'Inventory',
    'otherCurrentAssets': 'Other Current Assets',
    'totalCurrentAssets': 'Total Current Assets',
    'propertyPlantEquipmentNet': 'Property, Plant & Equipment (Net)',
    'goodwill': 'Goodwill',
    'intangibleAssets': 'Intangible Assets',
    'goodwillAndIntangibleAssets': 'Goodwill & Intangible Assets',
    'longTermInvestments': 'Long-Term Investments',
    'taxAssets': 'Tax Assets',
    'otherNonCurrentAssets': 'Other Non-Current Assets',
    'totalNonCurrentAssets': 'Total Non-Current Assets',
    'otherAssets': 'Other Assets',
    'totalAssets': 'Total Assets',
    'accountPayables': 'Accounts Payable',
    'shortTermDebt': 'Short-Term Debt',
    'taxPayables': 'Tax Payables',
    'deferredRevenue': 'Deferred Revenue',
    'otherCurrentLiabilities': 'Other Current Liabilities',
    'totalCurrentLiabilities': 'Total Current Liabilities',
    'longTermDebt': 'Long-Term Debt',
    'deferredRevenueNonCurrent': 'Deferred Revenue (Non-Current)',
    'deferredTaxLiabilitiesNonCurrent': 'Deferred Tax Liabilities (Non-Current)',
    'otherNonCurrentLiabilities': 'Other Non-Current Liabilities',
    'totalNonCurrentLiabilities': 'Total Non-Current Liabilities',
    'otherLiabilities': 'Other Liabilities',
    'capitalLeaseObligations': 'Capital Lease Obligations',
    'totalLiabilities': 'Total Liabilities',
    'preferredStock': 'Preferred Stock',
    'commonStock': 'Common Stock',
    'retainedEarnings': 'Retained Earnings',
    'accumulatedOtherComprehensiveIncomeLoss': 'Accumulated Other Comprehensive Income/Loss',
    'othertotalStockholdersEquity': 'Other Total Stockholders’ Equity',
    'totalStockholdersEquity': 'Total Stockholders’ Equity',
    'totalEquity': 'Total Equity',
    'totalLiabilitiesAndStockholdersEquity': 'Total Liabilities & Stockholders’ Equity',
    'minorityInterest': 'Minority Interest',
    'totalLiabilitiesAndTotalEquity': 'Total Liabilities & Total Equity',
    'totalInvestments': 'Total Investments',
    'totalDebt': 'Total Debt',
    'netDebt': 'Net Debt'
}

cashflow_names = {
    'netIncome': 'Net Income',
    'depreciationAndAmortization': 'Depreciation & Amortization',
    'deferredIncomeTax': 'Deferred Income Tax',
    'stockBasedCompensation': 'Stock-Based Compensation',
    'changeInWorkingCapital': 'Change in Working Capital',
    'accountsReceivables': 'Accounts Receivable',
    'inventory': 'Inventory',
    'accountsPayables': 'Accounts Payable',
    'otherWorkingCapital': 'Other Working Capital',
    'otherNonCashItems': 'Other Non-Cash Items',
    'netCashProvidedByOperatingActivities': 'Net Cash from Operating Activities',
    'investmentsInPropertyPlantAndEquipment': 'Investment in PP&E',
    'acquisitionsNet': 'Acquisitions (Net)',
    'purchasesOfInvestments': 'Purchases of Investments',
    'salesMaturitiesOfInvestments': 'Sales/Maturities of Investments',
    'otherInvestingActivites': 'Other Investing Activities',
    'netCashUsedForInvestingActivites': 'Net Cash from Investing Activities',
    'debtRepayment': 'Debt Repayment',
    'commonStockIssued': 'Common Stock Issued',
    'commonStockRepurchased': 'Common Stock Repurchased',
    'dividendsPaid': 'Dividends Paid',
    'otherFinancingActivites': 'Other Financing Activities',
    'netCashUsedProvidedByFinancingActivities': 'Net Cash from Financing Activities',
    'effectOfForexChangesOnCash': 'Effect of Forex Changes on Cash',
    'netChangeInCash': 'Net Change in Cash',
    'cashAtEndOfPeriod': 'Cash at End of Period',
    'cashAtBeginningOfPeriod': 'Cash at Beginning of Period',
    'operatingCashFlow': 'Operating Cash Flow',
    'capitalExpenditure': 'Capital Expenditure',
    'freeCashFlow': 'Free Cash Flow'
}

def explain(metric, df, prompt):
    json_data = df.to_json(orient='records')
    chat_session = st.session_state.chat_session
    response = chat_session.send_message(f"""
**Role:** Objective financial analyst focused on graph interpretation.

**Objective:** Analyze the {metric} data provided in JSON format: {json_data}, identify key trends visualized in the corresponding graph (based on the user's prompt), and explain these trends in the context of the prompt: "{prompt}".

**Task Instructions:**

1. **Understand Prompt:** Carefully consider the user's question: "{prompt}".

2. **Identify Trends:** Pinpoint significant trends (e.g., upward, downward, volatile) in the {metric} data visualized in the graph.

3. **Explain Graph Insights (Prompt Focused):**
   * **Key Findings:** Articulate the primary findings from the graph analysis, emphasizing trends relevant to the user's question: "{prompt}".
   * **Data Evidence:** Reference specific **numerical values** from the {metric} data to illustrate trends.
   * **Prompt Relevance:** Briefly explain how identified trends directly relate to and provide insights for the user's question: "{prompt}".

4. **Format Output:**
   * Use Markdown.
   * Structure your analysis under clear headers (e.g., `### Trend Analysis`, `### Key Observations`).
   * Maintain an objective tone.
   * When using dollar signs, use \$ to avoid unnecessary formatting.  

**Constraint Checklist:**
* [ ] Analysis based on trends and prompt?
* [ ] External knowledge *not* used?
* [ ] Markdown headers used?
* [ ] Explanation focuses on graph insights based on prompt?
* [ ] Output is understandable to users with moderate financial litteracy? 
                                         """)
    return response.text

def get_income_statements(tickers):
    income_statements = []

    for ticker in tickers:
        try:
            income_statement_df = fmp_api.get_income_statement(ticker)
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
            continue
        
        income_statement_df["ticker"] = ticker
        
        income_statements.append(income_statement_df)

    combined_income_statement_df = pd.concat(income_statements, ignore_index=True)

    return combined_income_statement_df

def get_balancesheet_statements(tickers):
    balancesheet_statements = []

    for ticker in tickers:
        try:
            balancesheet_statement_df = fmp_api.get_balance_sheet(ticker)
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
            continue
        
        balancesheet_statement_df["ticker"] = ticker
        
        balancesheet_statements.append(balancesheet_statement_df)

    combined_balancesheet_statement_df = pd.concat(balancesheet_statements, ignore_index=True)

    return combined_balancesheet_statement_df

def get_cashflow_statements(tickers):
    cashflow_statements = []

    for ticker in tickers:
        try:
            cashflow_statement_df = fmp_api.get_cashflow_statement(ticker)
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
            continue
        
        cashflow_statement_df["ticker"] = ticker
        
        cashflow_statements.append(cashflow_statement_df)

    combined_balancesheet_statement_df = pd.concat(cashflow_statements, ignore_index=True)

    return combined_balancesheet_statement_df

def trim_df(df, metric):
    return df[['ticker', 'date', metric]]

############# PAGE STARTS HERE #############

with st.sidebar:
    st.title(":green[EasyStock] Learner :chart:")
    
    st.session_state.selected_tickers = st.multiselect(
        "Select ticker:",
        api_config.get_ticker_options().keys(),
        default=st.session_state.selected_tickers,
        key="selectbox1",
        format_func=lambda x: api_config.get_ticker_options()[x],
        help = """Tickers are short symbols (like AAPL for Apple or MSFT for Microsoft) used to identify companies on the stock market.
                This tool only supports companies in the S&P 500.""",
        max_selections=3
    )

    selected_statement = st.selectbox(
        "Select statement to view: ",
        ['Income Statement', 'Balance Sheet Statement', 'Cashflow'],
        help="""
                Income Statement: Shows how much money a company made and spent over a period — revealing if it’s profitable.

                Balance Sheet: Shows what a company owns (assets), owes (liabilities), and what’s left for shareholders (equity) at a specific point in time.

                Cash Flow Statement: Tracks the actual flow of cash in and out of the company — helps show if the business can pay its bills and invest in growth.
                """,
        key='statement_selectbox',
    )       

    if selected_statement == 'Income Statement':
        metric_keys = list(income_statement_list.keys())

        selected_metric = st.selectbox(
            "Select metric to view:",
            metric_keys,
            format_func=lambda x: income_statement_list[x],
            key='metric_selectbox_is'
        ) 
        statment_df = get_income_statements(st.session_state.selected_tickers)
    elif selected_statement == 'Balance Sheet Statement':
        balance_sheet_keys = list(balance_sheet_list.keys())

        selected_metric = st.selectbox(
            "Select metric to view:",
            balance_sheet_keys,
            format_func=lambda x: balance_sheet_list[x],
            key='metric_selectbox_bs'
        )
        statment_df = get_balancesheet_statements(st.session_state.selected_tickers)
    elif selected_statement == 'Cashflow':
        cashflow_keys = list(cashflow_names.keys())

        selected_metric = st.selectbox(
            "Select metric to view:",
            cashflow_keys,
            format_func=lambda x: cashflow_names[x],
            key='metric_selectbox_cf'
        )
        statment_df = get_cashflow_statements(st.session_state.selected_tickers)

    selected_graph = st.selectbox(
        "Select graph type",
        ['Bar Chart', 'Line Chart'],
        help="""
            Bar charts are better for comparing values between categories (e.g., revenue across companies).
            Line charts are better for showing how a metric changes over time (e.g., profit growth over the years).     
            """,
        key='graph_selectbox',
    )

    with st.container(border=True):
        st.header("Links to other pages")
        st.page_link("tutorial.py", label="Tutorial")
        st.page_link("pages/1_🏠_Homepage.py", label="Dashboard")
        st.page_link("pages/4_News.py", label="News Analysis")
        st.page_link("pages/6_About.py", label="About")

st.title("Assisted Analysis")

main_col, r_col = st.columns((6,4), gap='small')

with main_col:
    # df = fin_plot.draw_eps_ratio(st.session_state.selected_tickers, finnhub_client)
    
    if selected_graph == 'Bar Chart':
        fig = px.bar(
                statment_df,
                x="date",
                y=selected_metric,
                color='ticker',
                title=f"{selected_metric} vs Time"
            )
    elif selected_graph == 'Line Chart':
        fig = px.line(
                statment_df,
                x="date",
                y=selected_metric,
                color='ticker',
                title=f"{selected_metric} vs Time"
            )
    
    st.plotly_chart(fig)
    
with r_col:
    inner_col = st.columns(3, vertical_alignment="bottom")

    with inner_col[0]:
        selected_task = st.selectbox(
            "Select a prompt",
            ['Explanation', 
             'Comparison between stocks',
             'Use my own prompt'],
            index=0,
            help="""
                Choose the type of question or task you’d like the AI to help with:\n
                • Explanation – Get a simple explanation of the selected financial data.\n
                • Comparison between stocks – Ask the AI to compare performance or metrics across different companies.\n
                • Use my own prompt – Write a custom question or request in your own words.\n
                """,
            key="task_selectbox"
        )
    
    if selected_task == 'Explanation':
        selected_explanation = st.selectbox(
            "Select a prompt",
            ['Briefly explain the graph', 
             'Further explain the graph', 
             'Explain what the metric means and how it affects a stock rating'],
            index=0,
            key="explanation_selectbox"
        )

        prompt_starter = selected_explanation

    elif selected_task == 'Comparison between stocks':
        col = st.columns(2)

        with col[0]:
            selected_stocks_compare = st.multiselect(
                "Select ticker:",
                st.session_state.selected_tickers,
                key="compare_stock_selectbox",
                max_selections=2
            )   

        with col[1]:
            selected_comparison = st.selectbox(
                "Select a prompt",
                ['Compare between these stocks', 
                 'Which stock is performing better'], # Add more questions
                index=0,
                key="compare_selectbox"
            )
        
        prompt_starter = f"{selected_comparison}. The stocks to compare are {', '.join(selected_stocks_compare)}."

    elif selected_task == 'Use my own prompt' :
        text_input = st.text_input(
            "Custom prompt",
            label_visibility='collapsed',
            placeholder="Enter prompt",
            key="text_input"
        )   

        prompt_starter = text_input

    with inner_col[1]:
        if st.button("Make prediction"):
            with inner_col[2]:  # Place spinner in the next column
                with st.spinner("Making prediction... Please wait", show_time=True):
                        st.session_state.llm_explaination = explain(selected_metric, trim_df(statment_df, selected_metric), prompt_starter)

    st.write(f"Prompt: {prompt_starter}")

    st.write(st.session_state.llm_explaination)

    # st.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
# Important metrics
#
# Return on equity
# Debt levels
# profit margins
# price to earning /
# price to book /
# revenuew growth
# earning growth
# analyst ratings
# news sentiment
