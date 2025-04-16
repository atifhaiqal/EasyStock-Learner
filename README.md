# EasyStock Learner

This project uses the power of Large Language Models (LLMs) to create an interactive and educational financial data visualization tool. It's designed to help users, even those with limited financial expertise, understand stock ratings and company performance through intuitive visualizations and LLM-driven insights.

**Key Objectives:**

* **Accessibility:** Provide an educational platform for users of all financial literacy levels.
* **Comprehensive Analysis:** Enable users to analyze company fundamentals, interpret sentiment from financial news, and understand stock ratings derived from machine learning.
* **Insightful Explanations:** Utilize LLMs to generate clear, natural-language explanations of complex financial data and stock ratings.

**Core Features:**

* **Multi-Stage LLM Pipeline:**
    * **Sentiment Analysis:** Employs FinBERT to analyze the sentiment of financial news articles.
    * **Financial Metric Interpretation:** Uses a general-purpose LLM to interpret financial data and generate human-readable explanations.
* **Interactive Visualizations:** Integrates a variety of charts and graphs to enable users to identify trends, correlations, and financial patterns from different angles.
* **Explanation Generators:** Provides LLM-powered summaries and interpretations of financial metrics, making them easier to understand.
* **Guided Tutorial Interface:** Offers a structured learning experience to help users navigate the tool and understand financial concepts.
* **Integrated News Viewer:** Allows users to access and contextualize market sentiment by exploring recent financial news relevant to companies.
* **Stock Rating System:** Uniquely combines financial news sentiment and structured company data to generate insightful stock ratings, providing users with a holistic understanding of the rationale behind each rating.

**Innovation:**

The core innovation of this project lies in its **seamless integration of qualitative (news sentiment) and quantitative (company fundamentals) data to generate stock ratings**. This combined analysis, coupled with comprehensive visualizations and LLM-powered explanations, offers users a powerful and accessible way to understand the complex world of stock analysis.

**Benefits for Users:**

* **Demystifies Financial Data:** Breaks down complex financial information into understandable insights.
* **Enhances Financial Literacy:** Provides an educational platform for learning about stock analysis and company performance.
* **Identifies Key Trends:** Empowers users to visually identify important financial patterns and correlations.
* **Contextualizes Information:** Integrates news sentiment to provide context for stock movements and company performance.
* **Provides Clear Explanations:** Leverages LLMs to explain the "why" behind stock ratings and financial metrics.
* **Accessible to All Levels:** Designed to be user-friendly for both beginners and those with some financial knowledge.

This project demonstrates the responsible and impactful application of LLMs in the financial domain, offering a dynamic and educational resource for anyone interested in understanding stock ratings and company performance.

## Application Requirements

The following Python libraries and their specified versions are required to run this application:

* altair==5.5.0
* finnhub_python==2.4.22
* numpy==2.2.4
* openai==1.75.0
* pandas==2.2.3
* Pillow==11.2.1
* plotly==5.22.0
* protobuf==6.30.2
* Requests==2.32.3
* streamlit==1.42.0
* streamlit_extras==0.6.0
* transformers==4.48.3
* yfinance==0.2.51

## Installation and Running Instructions

This section outlines the steps required to install and run the application. It is assumed that you have Python installed on your system.

**1. Clone the Repository or Download:**

First, you need to obtain the application code. You can either clone the Git repository or download the project files as a ZIP archive and extract them to your desired location.

**2. Install Requirements:**

The application relies on several Python libraries mentioned in [ Application Requirements](#application-requirements). To install these dependencies, navigate to the root directory of the cloned or downloaded project in your terminal or command prompt and run the following `pip` command:

```bash
pip install altair==5.5.0 finnhub_python==2.4.22 numpy==2.2.4 openai==1.75.0 pandas==2.2.3 Pillow==11.2.1 plotly==5.22.0 protobuf==6.30.2 Requests==2.32.3 streamlit==1.42.0 streamlit_extras==0.6.0 transformers==4.48.3 yfinance==0.2.51
```

**3. Run the Application:**

Once the requirements are successfully installed, you can launch the Streamlit application. Open your terminal or command prompt, navigate to the root directory of the project (where the main application file, named tutorial.py is located), and run the following command:

```bash
streamlit run tutorial.py
```

This command will start the Streamlit server, and your application should automatically open in your default web browser. You should now be able to interact with EasyStock Learner.

## Command reference

| Command           | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `streamlit run [rootpage]`            | Runs the app                        |
