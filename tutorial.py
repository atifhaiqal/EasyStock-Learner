import streamlit as st

############# PAGE CONFIG #############
st.set_page_config(
    page_title="EasyStock Learner",
    page_icon= "💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

############# PAGE STARTS HERE #############

with st.sidebar:
    st.title(":green[EasyStock] Learner :chart:")

    st.markdown("This page will help you understand the basics of this application. The tutorial is split into tabs to allow you to easily come back to it if needed.")

    st.markdown("Understand how the application works? Head over to the Dashboard to start using it!")
    
    with st.container(border=True):
        st.header("Links to other pages")
        st.page_link("pages/1_🏠_Homepage.py", label="Dashboard")
        st.page_link("pages/2_Financial Ratio Analysis.py", label="Assisted Analysis")
        st.page_link("pages/4_News.py", label="News Analysis")
        st.page_link("pages/6_About.py", label="About")

st.title("Tutorial Page")

welcome_tab, navigation_tab, dashboard_tab, analysis_tab, news_tab, about_tab = st.tabs(["Welcome", "Navigation", "Dashboard", "Assisted Analysis", "News", "About"])

with welcome_tab:
    st.header("Welcome to :green[EasyStock] Learner :chart:")
    st.image("assets/tutorial/dashboard.png", caption="EasyStock Learner Dashboard")

    st.header("What is :green[EasyStock] Learner?")
    st.markdown("EasyStock Learner is an educational platform powered by a Large Language Model (LLM), offering an interactive **sandbox** environment for users to develop a deeper understanding of investment principles.")
    st.markdown("EasyStock Learner is built for users with minimal to intermediate knowledge of investing and finance. Whether you’re just starting out or looking to sharpen your understanding, this tool allows you to explore financial concepts through interactive graphs and visualizations. With the support of machine learning models, it helps break down complex metrics into understandable insights.")
    st.markdown("All APIs used in the application are either free or operate under free-tier plans. The project is open-source, meaning you’re free to clone the original repository, explore the code, and modify it as you see fit.")
    st.markdown("**Note: This application is strictly for educational purposes only and is not intended to provide financial or investment advice.**")


with navigation_tab:
    st.header("Navigation")

    nav_col = st.columns([4,1,2])

    with nav_col[0]:
        st.markdown("Navigation is primarily handled through the sidebar, which allows users to easily access different sections of the application. In addition, contextual links may appear within the main interface when relevant. For instance, the news viewer on the homepage includes a link that directs users to the full news page for a more detailed view.")

    with nav_col[1]:
        st.image("assets/tutorial/navigation.png", caption="Navigation Bar")

    with nav_col[2]:
        st.image("assets/tutorial/news_viewer.png", caption="Dashboard News Viewer")
    

with dashboard_tab:
    st.header("Dashboard")

    dashboard_col1 = st.columns([2,2])

    with dashboard_col1[0]:
        st.markdown("""The main dashboard is where you’ll likely spend most of your time within EasyStock Learner.
                        It brings together key components that help you understand a stock’s overall performance and sentiment at a glance. These include:   
                            •	Consensus analyst ratings   
                            •	Current analyst rating chart   
                            •	A candlestick price chart (showing opening/closing prices and trade volume)   
                            •	A news sentiment chart with average sentiment     
                            •	The LLM-powered stock rating assistant   
                            •	A mini news viewer 
                            •	Stock perfomance widgets""")
        
        st.markdown("We’ll walk through each of these elements in more detail below.")

    with dashboard_col1[1]:
        st.image("assets/tutorial/dashboard.png", caption="EasyStock Learner Dashboard")

    st.divider()

    st.subheader("Selecting tickers")

    dashboard_col2 = st.columns([4,2,2])

    with dashboard_col2[0]:
        st.markdown("""Before any form of analysis can be done, a stock ticker must be selected first. 
                    The ticker selection box can be found in the sidebar.  
                    •	AAPL is selected by default  
                    •	Up to 3 tickers can be selected at one time.  
                    •	Users can search stock by manually typing in the selection box.  
                    •	Only SP500 tickers are available.""")
        

    with dashboard_col2[1]:
        st.image("assets/tutorial/ticker_selectbox.png", caption="Ticker selection box")

    with dashboard_col2[2]:
        st.image("assets/tutorial/ticker_selectbox_search.png", caption="Manually searching for ticker names")

    st.divider()

    st.subheader("Analyst ratings")

    dashboard_col3 = st.columns([4,2])

    with dashboard_col3[0]:
        st.markdown("""There are 2 parts to this visualization. The consensus ratings and the distribution of ratings of each selected stock""")

        st.markdown("#### Consensus rating")
        st.markdown("""
            This shows what most analysts think about a stock based on their average recommendations.

            - **Strong Buy** means most analysts are very confident in the stock.
            - **Buy** means it's generally seen as a good investment.
            - **Hold** means the stock is expected to stay steady — not a strong buy or sell.
            - **Sell** means analysts think the stock might go down.
            - **Strong Sell** means most analysts recommend selling or avoiding the stock.

            The closer the rating is to **Strong Buy**, the more positive the expert outlook is. For the most part, the best consensus a stock can have is **Buy** and the worst rating is **Sell** due to averaging out the ratings.
            """)
        
        with st.expander("See how its calculated"):
            st.markdown("""
                    The consensus rating is calculated using a weighted average of analyst recommendations, where each recommendation type is assigned a weight:

                    - **Strong Buy** = 5  
                    - **Buy** = 4  
                    - **Hold** = 3  
                    - **Sell** = 2  
                    - **Strong Sell** = 1

                    #### Formula:

                    $$
                    \\text{Consensus Score} = \\frac{
                    (5 \\times \\text{Strong Buy}) + 
                    (4 \\times \\text{Buy}) + 
                    (3 \\times \\text{Hold}) + 
                    (2 \\times \\text{Sell}) + 
                    (1 \\times \\text{Strong Sell})
                    }{
                    \\text{Total Number of Ratings}
                    }
                    $$

                    #### Interpretation:

                    | Consensus Score     | Label         |
                    |---------------------|---------------|
                    | **4.5 – 5.0**        | Strong Buy     |
                    | **3.5 – 4.49**       | Buy            |
                    | **2.5 – 3.49**       | Hold           |
                    | **1.5 – 2.49**       | Sell           |
                    | **1.0 – 1.49**       | Strong Sell    |

                    If there are no analyst ratings available, the score is not calculated.
                    """)

        st.markdown("#### Stock rating distribution")

        st.markdown("""
            This chart shows how analysts rate different stocks.

            Each group of bars represents a stock ticker (e.g., AAPL, MSFT), and each bar within the group shows how many analysts gave that stock a specific rating, such as **Strong Buy**, **Buy**, **Hold**, **Sell**, or **Strong Sell**.

            This lets you quickly compare how different stocks are viewed by analysts — for example, you can see which stocks have the most "Buy" ratings, or if there's a mix of opinions.

            Use this chart to spot trends and compare analyst sentiment across multiple stocks at once.
            """)

    with dashboard_col3[1]:
        st.image("assets/tutorial/consensus_rating.png", caption="Consensus rating of stock analysis")
        st.image("assets/tutorial/analyst_rating.png", caption="Distribution of analyst ratings")

    st.divider()

    st.subheader("Stock prices chart")

    dashboard_col4 = st.columns([2,2])

    with dashboard_col4[0]:
        st.markdown("""
            This chart shows how the stock prices has changed over time. It helps you understand the stock’s price history at a glance.

            You can toggle two features on and off using the toggles in the sidebar:

            - **Candlestick Chart**: This view shows the opening and closing prices, as well as the highest and lowest prices for each day. It’s a popular way to visualize daily price movement.
            - **Volume**: Adds bars at the bottom of the chart to show how many shares were traded each day. This helps you see how much interest or activity the stock had on a given date.

            By default, these toggles are turned on.     

            When the cursor is hovered over the chart, a tooltip will be shown. It displays information about the highlighted date, including the **opening and closing prices** and the **trading volume** for that day. The tooltip also helps you quickly see whether the **stock closed higher or lower** than it opened, giving you insight into the stock's daily movement.

            At the bottom of the chart, there's a **slider** that lets you scroll through the time range. You can use it to zoom in on specific periods or explore the stock's full history.

            To view the chart fullscreen, hover over the top-right of the chart where you can find a button to expand the chart. This is the same for every other chart in this application. 

            This chart is a great way to spot trends, patterns, and key moments in a stock’s performance.
            """)
        
    with dashboard_col4[1]:
        st.image("assets/tutorial/stock_price.png", caption="Stock price graph with opening and closing prices and volume of stock sold")

    stock_price_examples = st.columns(4)

    with stock_price_examples[0]:
        st.image("assets/tutorial/stock_settings.png", caption="Toggles for enabling/disabling the candlestick and trading volume. They can be found in the sidebar")
    
    with stock_price_examples[1]:
        st.image("assets/tutorial/stock_price_no_candlestick.png", caption="Stock price chart without candlestick for opening/closing prices")
    
    with stock_price_examples[2]:
        st.image("assets/tutorial/stock_price_no_volume.png", caption="Stock price chart without the trading volume")

    with stock_price_examples[3]:
        st.image("assets/tutorial/stock_price_median_only.png", caption="Stock price chart with median prices only")

    st.divider()

    st.subheader("News sentiment chart")

    dashboard_col5 = st.columns([4,2])

    with dashboard_col5[0]:
        st.markdown("""
            This section shows a breakdown of the latest news sentiment for each selected stock.

            A **sunburst chart** is used to visualize how news articles are categorized based on sentiment — **positive**, **neutral**, or **negative**. The sentiment analysis is performed using **FinBERT**, a machine learning model trained specifically for understanding financial news.

            For each selected ticker, the app analyzes the **20 most recent news articles**, giving you a quick overview of the tone of recent media coverage.

            You can also see an **average sentiment score** for each stock, helping you understand the general mood of the news — whether it's optimistic, cautious, or negative — at a glance.
            """)
        

    news_sentiment_examples = st.columns(4)

    with news_sentiment_examples[0]:
        st.image("assets/tutorial/news_sentiment.png", caption="News sentiment sunburst chart with the average news sentiment for each stock")

    with news_sentiment_examples[1]:
        st.image("assets/tutorial/news_sentiment_focused.png", caption="A focused view for the news sentiment for a specific stock (AAPL)")
    
    with news_sentiment_examples[2]:
        st.image("assets/tutorial/news_sentiment_invert.png", caption="An inverted sunburst chart to see the split between positive, neutral and negative news for all selected tickers")

    with news_sentiment_examples[3]:
        st.image("assets/tutorial/news_sentiment_settings.png", caption="Toggle for inverting the sunburt chart. It can be found in the sidebar")

    st.divider()

    st.subheader("Mini news viewer")
    
    dashboard_col7 = st.columns([4,2])

    with dashboard_col7[0]:
        st.markdown("""
            This section shows the **5 most recent news articles** for a selected stock ticker, giving you a quick snapshot of what’s currently being talked about in the markets.

            You can **choose which stock** you want to view news for using the dropdown menu. This makes it easy to stay updated on the latest headlines for any stock you're interested in.

            Clicking on the image or hyperlink brings you to the article.

            If you'd like to explore more, there's a **link at the bottom** that takes you to the full news page where you can dive deeper into recent articles.
            """)
        

    with dashboard_col7[1]:
        st.image("assets/tutorial/news_viewer.png", caption="Mini news viewer")

    st.divider()

    st.subheader("LLM-powered stock rating assistant")

    dashboard_col6 = st.columns([3,2])

    with dashboard_col6[0]:
        st.markdown("""
            This tool uses a **Large Language Model (LLM)** to help you understand how a stock is performing, based on financial data and, optionally, recent news sentiment.

            You can choose from two modes:

            - **Single Stock Rating**: Get a detailed breakdown for one stock. In this mode, you can only choose from the **three tickers** you initially selected earlier in the app.
            - **Comparative Rating**: Compare up to **two stocks side by side**, helping you decide which one may be performing better or showing stronger potential.

            There’s also an option to **include or exclude news sentiment** from the prediction. Enabling news allows the model to factor in recent media coverage, while disabling it makes the rating based solely on financial metrics.
                    
            After the settings have been chosen, press the **"Make prediction"** button to query the LLM. You’ll see a spinning progress indicator while the model processes your request.

            This assistant gives you a human-readable explanation along with the model’s prediction, so you can learn more about what drives the rating. You may need to scroll to read the full text. 
            """)
        
        with st.expander("What to Expect from the Stock Rating Output"):
            st.markdown("""
                After submitting your request, the assistant will generate a detailed, easy-to-read analysis based on the selected stock(s). Here's what you'll see:

                - **A Clear Recommendation**: The model provides a straightforward rating such as **Buy**, **Hold**, or **Sell**.

                - **Key Factors Driving the Rating**: The model explains the reasoning behind the recommendation by highlighting the most relevant and impactful factors. These may include:
                - **Recent News Sentiment**: If enabled, the model will analyze the latest news headlines using FinBERT and identify dominant themes (e.g. negative outlooks, policy risks, or product announcements). It specifically calls out **which topics or headlines** are currently influencing the rating.
                - **Stock Price Trends**: Insight into recent price performance over various timeframes (5-day, 13-week, YTD, etc.).
                - **Valuation Metrics**: P/E and P/B ratios are considered to assess whether the stock may be over- or under-valued.
                - **Profitability and Growth**: The model checks for changes in earnings and overall financial health.
                - **Liquidity**: Key indicators like the current ratio help assess the company’s ability to meet short-term obligations.

                - **Natural Language Summary**: The assistant breaks down technical data into a beginner-friendly explanation, so you can learn as you go.

                - **Optional News Impact Toggle**: If you choose to include news in your analysis, the assistant will weigh recent headlines into its evaluation and specifically mention **how recent events are affecting the stock** — providing context to the sentiment trends.

                - **⚠️ Disclaimer**: This tool is for **educational purposes only** and is not intended as financial advice. Please do your own research or speak to a licensed financial advisor before making any investment decisions.

                This feature is designed to help you understand how both **financial data and current news** can influence stock ratings — making it a powerful learning tool for beginner to intermediate users.
                """)
    
    with dashboard_col6[1]:
        st.image("assets/tutorial/llm_stock_rating.png", caption="LLM stock rating for AAPL")

    llm_rating_examples = st.columns(4)

    with llm_rating_examples[0]:
        st.image("assets/tutorial/llm_stock_rating_options.png", caption="Selection box between single stock or comparative rating")

    with llm_rating_examples[1]:
        st.image("assets/tutorial/llm_stock_rating_options2.png", caption="Selection box for the stocks to rate. When doing comparative rating, users can pick up to two stocks")
    
    with llm_rating_examples[2]:
        st.image("assets/tutorial/llm_stock_rating_options3.png", caption="Selection box to include or exclude news information in the rating")

    with llm_rating_examples[3]:
        st.image("assets/tutorial/llm_stock_rating_spinner.png", caption="Spinning progress bar")


    st.divider()

    st.subheader("Stock Perfomance Overview")

    dashboard_col8 = st.columns([2,2])

    with dashboard_col8[0]:
        st.markdown("""
            The **Stock Performance Widgets** offer a quick, interactive way to track key financial metrics for the selected stock. These widgets display important data points such as:

            - **Revenue**
            - **P/E (Price-to-Earnings) Ratio**
            - **P/B (Price-to-Book) Ratio**
            - **EBITDA** and more

            Each widget shows the **percentage growth or decline** for the selected metric over a customizable time period. You can easily adjust the **time range** using the menu in the top-right corner, choosing between a 1-year or 5-year period.

            This allows you to monitor how a stock has been performing over different periods and identify trends in key financial metrics at a glance.

            At the bottom of the section, there is a **link to the assisted analysis page** where you can dive deeper into the model’s breakdown of the stock and get further insights into its performance.

            These widgets are designed to give you a quick snapshot of the stock’s performance without needing to dig through complex data. It’s an excellent way to get a high-level overview of the stock's financial health.
                    
            In the event that the data is unavailable, it will be displayed  as "unavailable".
            """)

    with dashboard_col8[1]:
        st.image("assets/tutorial/stock_performance.png", caption="Stock performance widget for AAPL, NVDA and MSFT")

with analysis_tab:
    st.header("Assited Analysis")

    assisted_analysis = st.columns([3,2])

    with assisted_analysis[0]:
        st.markdown("""
            The **Assisted Analysis Page** is where you can dive deeper into financial metrics and get a more detailed breakdown of your selected stocks. Here’s what to expect:

            - **Persisted Tickers**: The tickers you chose earlier on the dashboard will automatically **transfer** to this page, so you can continue working with the same set of stocks without needing to reselect them.

            - **Metrics Selection**: You’ll have the ability to **choose metrics** from various financial statements, including:
                - **Income Statement**
                - **Balance Sheet**
                - **Cash Flow**

                This allows you to focus on the specific data points that matter most to your analysis. You can also decide which type of **graph** you want to plot the data as, either as a line chart or a bar graph.

            - **LLM Prompts**: You can ask the LLM for insights into the stocks you're analyzing. There are pre-made prompts for:
                - **Single Stock Analysis**
                - **Comparative Stock Analysis**
            
                Alternatively, you can **write your own custom prompt** to get tailored insights. For example, you can query the model about specific financial metrics or trends you're interested in. The prompt is tied to the metric you are currently visualizing, therefore make sure to ask questions that relate to that metric.

            - **Making Predictions**: To get a stock prediction or analysis, simply press the **"Make Prediction"** button. This will query the LLM, and a **spinning progress bar** will show as it processes your request.

            Feel free to explore the various options on the page and experiment with the different graphing and querying options. The **Assisted Analysis Page** is a powerful tool for digging deeper into a stock’s performance and getting customized insights based on your needs.
            """)
        
    with assisted_analysis[1]:
        st.image("assets/tutorial/assisted_analysis.png", caption="Assisted analysis page")

        assisted_analysis_inner = st.columns(2)

        with assisted_analysis_inner[0]:
            st.image("assets/tutorial/assisted_analysis_tickers.png", caption="Ticker selection box")
        
        with assisted_analysis_inner[1]:
            st.image("assets/tutorial/assisted_analysis_settings.png", caption="Metric plotting options")

        st.image("assets/tutorial/assisted_analysis_prompt.png", caption="Example of prompts available to user")

        st.image("assets/tutorial/assisted_analysis_self_prompt.png", caption="Input box for custom prompts")


with news_tab:
    st.header("News")

with about_tab:
    st.header("About")