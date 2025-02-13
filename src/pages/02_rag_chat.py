import streamlit as st

import os
from dotenv import load_dotenv
import pandas as pd
import json


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
from utils.prompts import rag_prompt

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
vectorstore = Chroma(
    client=client, collection_name=CHROMA_COLLECTION_NAME, embedding_function=embedding_model
)
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)
retriever = vectorstore.as_retriever(k=10)

render_chat_messages(st.session_state.messages)

if prompt := st.chat_input("Message to LLM"):
    # display user message
    with st.chat_message("user"):
        st.write(prompt)
        prompt_embedding = embedding_model.embed_query(prompt)
    # add message to history
    st.session_state.messages.append({"role": "user", "text": prompt})

    # retrieve documents, convert as data frame
    retrieved_docs = retriever.invoke(prompt)
    retrieved_docs_dict = []
    for doc in retrieved_docs:
        retrieved_docs_dict.append(
            {
                'path': doc.metadata.get('file'),
                'chunk_id': doc.metadata.get('chunk_id'),
                'content': doc.page_content,
            }
        )
    query_df = pd.DataFrame(retrieved_docs_dict)
    query_df = query_df.reset_index().rename(columns={'index': 'ref_id'})
    query_df['ref_id'] += 1

    # generate response and display it
    with st.chat_message("assistant"):
        old_messages = st.session_state.messages[1:]

        rag_prompt_formatted = rag_prompt.format(
            question=st.session_state.messages[-1]['text'],
            context=json.dumps(query_df.to_dict(orient='records'), indent=2),
        )

        rag_col, ref_col = st.columns(2)
        with rag_col:
            with st.popover("RAG Prompt"):
                st.write(rag_prompt_formatted)

        with ref_col:
            with st.popover("Reference Documents"):
                st.dataframe(query_df)

        stream_result = ""

        def stream_data():
            global stream_result
            messages = generate_message_history([{'role': 'user', 'text': rag_prompt_formatted}])
            for chunk in llm.stream(messages):
                stream_result += chunk.content
                yield chunk

        st.write_stream(stream_data)

    # add message to history
    st.session_state.messages.append({"role": "ai", "text": stream_result})
