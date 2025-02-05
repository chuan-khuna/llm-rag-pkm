import streamlit as st

import os
from dotenv import load_dotenv
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(CURRENT_DIR, '..')
load_dotenv(dotenv_path=os.path.join(APP_DIR, '..', '.env'))

RAG_DATA_DIR = os.path.join(APP_DIR, '..', 'rag-data')


from utils.file_utils import list_all_files, split_readable_file_path
from utils.chroma import get_chroma_client
from utils.streamlit_conf import *
from utils.llm_agent import OllamaAgent


client = get_chroma_client(
    os.environ.get('CHROMA_HOST', 'localhost'),
    int(os.environ.get('CHROMA_PORT', '8200')),
    os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS', 'password'),
)

# initalise state
if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    llm_model_name = st.selectbox("Select LLM Model", LLM_CHOICES, index=DEFAULT_LLM_IDX)
    temperature = st.slider("Temperature", 0.1, 1.0, 1.0, 0.1)
    llm_agent = OllamaAgent(llm_model_name, temperature)


with st.spinner('Loading data...'):
    md_files = list_all_files(RAG_DATA_DIR, '.md')

df = pd.DataFrame(md_files, columns=['abs_path'])
df['path'] = df['abs_path'].apply(
    lambda x: split_readable_file_path(split_by=os.path.join(APP_DIR, '..'), file_path=x)
)
