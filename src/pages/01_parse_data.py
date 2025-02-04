import streamlit as st
import os

from utils.file_utils import list_all_files

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(CURRENT_DIR, '..')
RAG_DATA_DIR = os.path.join(APP_DIR, '..', 'rag-data')

with st.spinner('Loading data...'):
    md_files = list_all_files(RAG_DATA_DIR, '.md')

readable_md_files = list(
    map(lambda file_path: file_path.split(os.path.join(APP_DIR, '..'))[1][1:], md_files)
)
