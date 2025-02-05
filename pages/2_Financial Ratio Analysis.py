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
    page_title="Financial Ratio Analysis",
)

# Initialising global variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""

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

st.title("Financial Ratio Analysis")

selectedTickers = st.multiselect(
    "Select ticker:",
    api_config.get_ticker_options(),
    default=['MSFT', 'GOOGL'],
    key="selectbox1"
)

fin_plot.draw_pe_ratio(selectedTickers, finnhub_client)
fin_plot.draw_pb_ratio(selectedTickers, finnhub_client)


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
