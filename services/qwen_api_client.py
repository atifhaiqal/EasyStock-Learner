from pickle import NONE
from openai import OpenAI
import pandas as pd
import streamlit as st
import os

class Qwen_APIClient:
    @st.cache_resource
    def get_qwen_client(api_key):
        """Streamlit cached resource for the FMP API client."""
        return Qwen_APIClient(api_key)

    def __init__(self, api_key):
        self.client = OpenAI(
            # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
            api_key= api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    def describe_pe_ratio(self, pe_ratio):
        pe_data_json = pe_ratio.to_json(orient="records", indent=4)
        message = """Analyse the performance and provide a concise insights but avoid introductions.
                    Look over the trends over the years, highlighting key growth"""

        completion = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': 'You are a an assistant in a investment learning app called EasyStock Learner. Your role is to help interpret the importance table of an XGBoost model that predicts the stock rating of any given stock and give the reasoning behind the rating (either buy or sell). Apart from that your role is also to help users understand the meaning behind the different metrics and ratios in finance. You are required to analyse time series financial data and generate an explanation of the performance of the metric. Be concise and start directly with insights.'},
                {'role': 'user', 'content': f"Here is a list of companies and their P/E ratios:\n\n{pe_data_json}\n\n. {message}"}
                ]
        )
        if(completion.choices[0].message.content != NONE):
            st.write(completion.choices[0].message.content)
            print(completion.choices[0].message.content)
        else:
            print("QWEN Error: No message returned")
