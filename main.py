# from transformers import pipeline
import streamlit as st
import requests
import os
import dotenv

dotenv.load_dotenv()

# Local inference
# pipeline_en = pipeline('translation', 'Helsinki-NLP/opus-mt-en-vi')
# pipeline_vi = pipeline('translation', 'Helsinki-NLP/opus-mt-vi-en')

# Use cloud-based inference API due to Streamlit sharing's limitation
headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

def query(api_url, text):
	response = requests.post(api_url, headers=headers, json={ 'inputs': text })
	return response.json()

def pipeline_en(text):
     return query("https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-vi", text)

def pipeline_vi(text):
     return query("https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-vi-en", text)
	
st.title('English -> Tiếng Việt')

def generate_response(text, pipeline):
    try:
        response = pipeline(text)
        st.info(response[0]['translation_text'])
    except:
        st.error('Oops! Something went wrong. Please try again.')

with st.form('en_vi'):
    text = st.text_area('Enter text:', 'Today is a good day.')
    submitted = st.form_submit_button('Translate')
    if submitted:
        generate_response(text, pipeline_en)

st.title('Tiếng Việt -> English')
with st.form('vi_en'):
    text = st.text_area('Enter text:', 'Chào bạn.')
    submitted = st.form_submit_button('Dịch')
    if submitted:
        generate_response(text, pipeline_vi)