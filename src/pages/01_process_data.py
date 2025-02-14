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


from langchain_community.document_loaders import PyPDFLoader

from utils.chroma import get_chroma_client
from utils.streamlit_conf import (
    CHROMA_COLLECTION_NAME,
    EMBEDDING_CHOICES,
    DEFAULT_EMBEDDING_IDX,
    LLM_CHOICES,
    DEFAULT_LLM_IDX,
)
from utils.file_utils import list_all_files, split_readable_file_path

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


llm_model_name, temperature, embedding_model_name = sidebar(client, hide_delete_collection=False)

with st.spinner('Loading data...'):
    md_files = list_all_files(RAG_DATA_DIR, '.md')
    pdf_files = list_all_files(RAG_DATA_DIR, '.pdf')

files = md_files + pdf_files
df = pd.DataFrame(files, columns=['abs_path'])
df['path'] = df['abs_path'].apply(
    lambda x: split_readable_file_path(split_by=os.path.join(APP_DIR, '..'), file_path=x)
)

st.dataframe(df)

# intialise LLM and embedding models
llm = ChatOllama(model=llm_model_name, temperature=temperature)
llm_json_mode = ChatOllama(model=llm_model_name, temperature=temperature, format="json")
embedding_model = OllamaEmbeddings(model=embedding_model_name)
vectorstore = Chroma(
    client=client, collection_name=CHROMA_COLLECTION_NAME, embedding_function=embedding_model
)
markdown = markdown_textsplitter.MarkdownTextSplitter()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)

collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)


def index_markdown(abs_path, ref_path, vectorstore):
    with open(abs_path, 'r') as f:
        content = f.read()
        chunks = markdown.split_text(content)

        st.toast(f'Processing `{ref_path}` ({len(chunks)} chunks)')

        for chunk_idx, chunk_content in enumerate(chunks):
            metadata = {'file': ref_path, 'chunk_id': chunk_idx + 1}
            chunk_id = f'{ref_path}_{chunk_idx + 1}'

            vectorstore.delete(ids=[chunk_id])
            doc_obj = Document(
                page_content=chunk_content,
                metadata={'file': ref_path, 'chunk_id': chunk_idx + 1},
                id=chunk_id,
            )

            vectorstore.add_documents(documents=[doc_obj], ids=[chunk_id])


def index_pdf(abs_path, ref_path, vectorstore):
    loader = PyPDFLoader(abs_path)
    pages = []

    for page in loader.lazy_load():
        pages.append(page)

    st.toast(f'Processing `{ref_path}` ({len(pages)} pages)')

    splitted_chunks = text_splitter.split_documents(pages)

    for chunk_idx, chunk in enumerate(splitted_chunks):
        page_number = chunk.metadata.get('page_number', -1)
        metadata = {'file': ref_path, 'chunk_id': chunk_idx, 'page_number': page_number}
        chunk_id = f'{ref_path}_{page_number}_{chunk_idx}'
        content = chunk.page_content

        vectorstore.delete(ids=[chunk_id])
        doc_obj = Document(page_content=content, metadata=metadata, id=chunk_id)
        vectorstore.add_documents(documents=[doc_obj], ids=[chunk_id])


if st.button('Process data'):

    for _, row in df.iterrows():
        abs_path = row['abs_path']
        ref_path = row['path']

        if abs_path.endswith('.md'):
            index_markdown(abs_path, ref_path, vectorstore)
        elif abs_path.endswith('.pdf'):
            index_pdf(abs_path, ref_path, vectorstore)
