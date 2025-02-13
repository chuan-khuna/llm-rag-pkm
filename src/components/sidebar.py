import streamlit as st

from utils.streamlit_conf import (
    CHROMA_COLLECTION_NAME,
    EMBEDDING_CHOICES,
    DEFAULT_EMBEDDING_IDX,
    LLM_CHOICES,
    DEFAULT_LLM_IDX,
)


def sidebar(chroma_client, hide_delete_collection=True):
    with st.sidebar:
        llm_model_name = st.selectbox("Select LLM Model", LLM_CHOICES, index=DEFAULT_LLM_IDX)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

        embedding_model_name = st.selectbox(
            "Select Embedding Model", EMBEDDING_CHOICES, index=DEFAULT_EMBEDDING_IDX
        )

        st.write('---')

        with st.spinner('Loading Chroma collection...'):
            collection = chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
            # get number of documents in collection
            num_docs = collection.count()
            st.write(f'Number of documents in collection: {num_docs}')

            if not hide_delete_collection:
                if st.button('Delete collection'):
                    with st.spinner('Deleting Chroma collection...'):
                        chroma_client.delete_collection(CHROMA_COLLECTION_NAME)

    return llm_model_name, temperature, embedding_model_name
