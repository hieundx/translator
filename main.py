from transformers import pipeline
import streamlit as st

pipeline_en = pipeline('translation', 'Helsinki-NLP/opus-mt-en-vi')
pipeline_vi = pipeline('translation', 'Helsinki-NLP/opus-mt-vi-en')


st.title('English -> Tiếng Việt')

def generate_response(input_text, pipeline):
    response = pipeline(input_text)

    st.info(response[0]['translation_text'])

with st.form('en_vi'):
    text = st.text_area('Enter text:', 'Today is a good day.')
    submitted = st.form_submit_button('Translate')
    if submitted:
        generate_response(text, pipeline_en)

st.title('Tiếng Việt -> English')
with st.form('vi_en'):
    text = st.text_area('Enter text:', 'Chào bạn.')
    submitted = st.form_submit_button('Translate')
    if submitted:
        generate_response(text, pipeline_vi)