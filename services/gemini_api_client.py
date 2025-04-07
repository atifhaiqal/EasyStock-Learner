from pickle import NONE
from openai import OpenAI
import pandas as pd
import streamlit as st
import os
from google import genai

class Gemini_APIClient:
    # genai.configure(api_key=os.environ["AIzaSyDeEKkLTe_Gbv0jTn4Ormx5OUy8cuz8ahA"])

    #         # Create the model
    # generation_config = {
    #     "temperature": 1,
    #     "top_p": 0.95,
    #     "top_k": 40,
    #     "max_output_tokens": 1024,
    #     "response_mime_type": "text/plain",
    # }

    # model = genai.GenerativeModel(
    #     model_name="gemini-1.5-pro",
    #     generation_config=generation_config,
    #     system_instruction="You are a an assistant in a investment learning app called EasyStock Learner. Your role is to predict the stock rating of a company given its financial data and give the reasoning behind the rating (either buy or sell). Apart from that your role is also to help users understand the meaning behind the different metrics and ratios in finance. You are required to analyse time series financial data and generate an explanation of the performance of the metric. Be concise and start directly with insights.",
    # )

        

    # def describe_pe_ratio(self, pe_ratio):
        # pe_data_json = pe_ratio.to_json(orient="records", indent=4)
        # message = """Analyse the performance and provide a concise insights but avoid introductions.
        #             Look over the trends over the years, highlighting key growth"""

        # completion = self.client.chat.completions.create(
        #     model="qwen-plus",
        #     messages=[
        #         {'role': 'system', 'content': 'You are a an assistant in a investment learning app called EasyStock Learner. Your role is to help interpret the importance table of an XGBoost model that predicts the stock rating of any given stock and give the reasoning behind the rating (either buy or sell). Apart from that your role is also to help users understand the meaning behind the different metrics and ratios in finance. You are required to analyse time series financial data and generate an explanation of the performance of the metric. Be concise and start directly with insights.'},
        #         {'role': 'user', 'content': f"Here is a list of companies and their P/E ratios:\n\n{pe_data_json}\n\n. {message}"}
        #         ]
        # )
        # if(completion.choices[0].message.content != NONE):
        #     st.write(completion.choices[0].message.content)
        #     print(completion.choices[0].message.content)
        # else:
        #     print("QWEN Error: No message returned")

    # def make_prediction(self, financial_data, news_sentiment):
    #     financial_data_json = financial_data.to_json(orient="records", indent=4)
    #     message = f"""Here is the financial data of a given company:\n\n{financial_data_json}\n\n. 
    #                 The sentiment analysis of 20 of the most recent news surrounding the company is {news_sentiment}.
    #                 Given these data points, make a stock rating on the given company (buy, hold, or sell). 
    #                 Analyse the performance and provide a concise insights but avoid introductions.
    #                 Highlight key reasonings as to why the decision has been made. 
    #                 Give the results in markdown format, with headers for key reasons for the rating of the stock. 
    #                 """

    #     response = self.chat_session.send_message(f"{message}")

    #     return response.text

@st.cache_resource
def get_gemini_client():
    """Streamlit cached resource for the Qwen client."""
    return Gemini_APIClient()








