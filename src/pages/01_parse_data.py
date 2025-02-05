import streamlit as st
import os
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(CURRENT_DIR, '..')
load_dotenv(dotenv_path=os.path.join(APP_DIR, '..', '.env'))

RAG_DATA_DIR = os.path.join(APP_DIR, '..', 'rag-data')


from utils.file_utils import list_all_files, split_readable_file_path
from utils.chroma import get_chroma_client


with st.spinner('Loading data...'):
    md_files = list_all_files(RAG_DATA_DIR, '.md')


readable_md_files = list(
    map(
        lambda file_path: split_readable_file_path(
            split_by=os.path.join(APP_DIR, '..'), file_path=file_path
        ),
        md_files,
    )
)


client = get_chroma_client(
    os.environ.get('CHROMA_HOST', 'localhost'),
    int(os.environ.get('CHROMA_PORT', '8200')),
    os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS', 'password'),
)
