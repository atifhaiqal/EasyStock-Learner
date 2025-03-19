import streamlit as st
from config.api_settings import FMP_APIConfig
import finnhub

# importing api clients
from config.api_settings import FMP_APIConfig
from services.fmp_api_client import FMP_APIClient, get_fmp_client
from services.alphavantage_api_client import AlphaVantage_APIClient, get_alphavantage_client

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

#temp
FMP_API_KEY = "OSrMm0u3iB8mz1iJMaK0XQno7DyqQKRw"
AV_API_KEY = 'WGHKWKAR5TGFV4IC'
FINNHUB_API_KEY = 'ctkp081r01qn6d7j5lt0ctkp081r01qn6d7j5ltg'

fmp_api = get_fmp_client(FMP_API_KEY)
av_api = get_alphavantage_client(AV_API_KEY)
finnhub_client = finnhub.Client(FINNHUB_API_KEY)
api_config = FMP_APIConfig()

fin_plot = Finnhub_Plot_Components()
fmp_plot = FMP_Plot_Components()
av_plot = AlphaVantage_Plot_Components()

############# PAGE STARTS HERE #############

with st.sidebar:
    st.title(":green[EasyStock] Learner :chart:")
    
    st.session_state.selected_tickers = st.multiselect(
        "Select ticker:",
        api_config.get_ticker_options().keys(),
        default=st.session_state.selected_tickers,
        key="selectbox1",
        format_func=lambda x: api_config.get_ticker_options()[x],
        max_selections=3
    )       
    
    with st.container(border=True):
        st.header("Links to other pages")
        st.page_link("1_🏠_Homepage.py", label="Dashboard")
        st.page_link("pages/4_News.py", label="News Analysis")
        st.page_link("pages/6_About.py", label="About")

st.title("Assisted Analysis")

main_col, r_col = st.columns((6,4), gap='small')

with main_col:
    fmp_plot.draw_net_income(st.session_state.selected_tickers, fmp_api)

with r_col:
    selected_task = st.selectbox(
        "Select a prompt",
        ['Explanation', 'Comparison between stocks'],
        index=0,
        key="task_selectbox"
    )

    if selected_task == 'Explanation':
        selected_explanation = st.selectbox(
            "Select a prompt",
            ['Briefly explain the graph', 'Further explain the graph', 'Explain what the metric means and how it affects a stock rating'],
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
                ['Compare between these stocks', 'Which stock is performing better'], # Add more questions
                index=0,
                key="compare_selectbox"
            )
        
        prompt_starter = f"{selected_comparison}. The stocks to compare are {', '.join(selected_stocks_compare)}."

    st.write(prompt_starter)

    st.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
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
