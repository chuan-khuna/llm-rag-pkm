import streamlit as st

import os
from dotenv import load_dotenv
import pandas as pd


from utils.file_utils import list_all_files, split_readable_file_path
from utils.chroma import get_chroma_client
from utils.streamlit_conf import *
from utils.llm_agent import OllamaAgent

from langchain_text_splitters import markdown as markdown_textsplitter
from langchain_ollama import OllamaEmbeddings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(CURRENT_DIR, '..')
load_dotenv(dotenv_path=os.path.join(APP_DIR, '..', '.env'))

RAG_DATA_DIR = os.path.join(APP_DIR, '..', 'rag-data')


client = get_chroma_client(
    os.environ.get('CHROMA_HOST', 'localhost'),
    int(os.environ.get('CHROMA_PORT', '8200')),
    os.environ.get('CHROMA_SERVER_AUTHN_CREDENTIALS', 'password'),
)


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

        if st.button('Delete collection'):
            with st.spinner('Deleting Chroma collection...'):
                client.delete_collection(CHROMA_COLLECTION_NAME)


with st.spinner('Loading data...'):
    md_files = list_all_files(RAG_DATA_DIR, '.md')

df = pd.DataFrame(md_files, columns=['abs_path'])
df['path'] = df['abs_path'].apply(
    lambda x: split_readable_file_path(split_by=os.path.join(APP_DIR, '..'), file_path=x)
)

embedding_model = embedding_model = OllamaEmbeddings(model=embedding_model_name)
splitter = markdown_textsplitter.MarkdownTextSplitter()


if st.button('Process data'):

    for _, row in df.iterrows():
        abs_path = row['abs_path']
        ref_path = row['path']

        st.toast(f'Processing {ref_path} ...')

        with open(abs_path, 'r') as f:
            content = f.read()
            chunks = splitter.split_text(content)

            for chunk_idx, chunk in enumerate(chunks):
                metadata = {'file': ref_path, 'chunk_id': chunk_idx + 1}
                doc = chunk
                embedding = embedding_model.embed_query(content)
                chunk_id = f'{ref_path}_{chunk_idx + 1}'

                collection.delete(ids=[chunk_id])

                collection.add(
                    documents=[doc], embeddings=[embedding], metadatas=[metadata], ids=[chunk_id]
                )
