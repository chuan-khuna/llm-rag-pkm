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

from langchain_ollama import OllamaEmbeddings

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
    embedding_model_name = st.selectbox(
        "Select Embedding Model", EMBEDDING_CHOICES, index=DEFAULT_EMBEDDING_IDX
    )

    st.write('---')

    with st.spinner('Loading Chroma collection...'):
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        # get number of documents in collection
        num_docs = collection.count()
        st.write(f'Number of documents in collection: {num_docs}')


def render_chat_messages(messages):
    for message in messages:
        role = message['role']
        with st.chat_message(role):
            st.write(message['text'])


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
            for chunk in llm_agent.stream(st.session_state.messages):
                global stream_result
                stream_result += chunk.content
                yield chunk

        st.write_stream(stream_data)

    st.session_state.messages.append({"role": "ai", "text": stream_result})
