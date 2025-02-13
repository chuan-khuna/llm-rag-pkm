import streamlit as st

import os
from dotenv import load_dotenv
import pandas as pd


from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import markdown as markdown_textsplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


from utils.chroma import get_chroma_client
from utils.streamlit_conf import (
    CHROMA_COLLECTION_NAME,
    EMBEDDING_CHOICES,
    DEFAULT_EMBEDDING_IDX,
    LLM_CHOICES,
    DEFAULT_LLM_IDX,
)
from utils.llm_utils import generate_message_history

from components.sidebar import sidebar


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(CURRENT_DIR, '..')
load_dotenv(dotenv_path=os.path.join(APP_DIR, '..', '.env'))


RAG_DATA_DIR = os.path.join(APP_DIR, '..', 'rag-data')

client = get_chroma_client(
    os.environ.get('CHROMA_HOST', 'localhost'),
    int(os.environ.get('CHROMA_PORT', '8200')),
    os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS', 'password'),
)

# initalise state
if 'messages' not in st.session_state:
    st.session_state.messages = []

llm_model_name, temperature, embedding_model_name = sidebar(client)


def render_chat_messages(messages):
    for message in messages:
        role = message['role']
        with st.chat_message(role):
            st.write(message['text'])


# intialise LLM and embedding models
llm = ChatOllama(model=llm_model_name, temperature=temperature)
llm_json_mode = ChatOllama(model=llm_model_name, temperature=temperature, format="json")
embedding_model = OllamaEmbeddings(model=embedding_model_name)
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)
render_chat_messages(st.session_state.messages)


if prompt := st.chat_input("Message to LLM"):
    # display user message
    with st.chat_message("user"):
        st.write(prompt)
    # add message to history
    st.session_state.messages.append({"role": "user", "text": prompt})

    with st.chat_message("assistant"):
        stream_result = ""

        def stream_data():
            global stream_result
            messages = generate_message_history(st.session_state.messages)
            for chunk in llm.stream(messages):
                stream_result += chunk.content
                yield chunk

        st.write_stream(stream_data)

    st.session_state.messages.append({"role": "ai", "text": stream_result})
