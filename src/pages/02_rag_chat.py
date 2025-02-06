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


embedding_model = OllamaEmbeddings(model=embedding_model_name)


render_chat_messages(st.session_state.messages)

if prompt := st.chat_input("Message to LLM"):
    # display user message
    with st.chat_message("user"):
        st.write(prompt)
        prompt_embedding = embedding_model.embed_query(prompt)
    # add message to history
    st.session_state.messages.append({"role": "user", "text": prompt})

    query_res = collection.query(query_embeddings=[prompt_embedding], n_results=10)
    query_df = pd.DataFrame(
        {
            'metadata': query_res['metadatas'][0],
            'document': query_res['documents'][0],
            'distance': query_res['distances'][0],
        }
    )
    query_df['path'] = query_df['metadata'].apply(lambda x: x['file'])
    query_df['chunk_id'] = query_df['metadata'].apply(lambda x: x['chunk_id'])
    query_df.drop(columns=['metadata'], inplace=True)
    query_df = query_df.reset_index().rename(columns={'index': 'ref_id'})
    query_df['ref_id'] += 1

    # generate response and display it
    with st.chat_message("assistant"):
        old_messages = st.session_state.messages[1:]

        rag_prompt = f"""
User question: {st.session_state.messages[-1]['text']}

Here are the references that I found in the database:


<DOCUMENTS>

```
{query_df.to_markdown(index=False)}
```

</DOCUMENTS>


<INSTRUCTIONS>
You must provide answer to the user question based on the DOCUMENTS provided above.

When you generate answer from the DOCUMENTS.
You must provide the reference id to the source. 
Use IEEE format, for example <your answer from the DOCUMENT> [<ref_id>].
</INSTRUCTIONS>
"""
        with st.expander("RAG Prompt"):
            st.write(rag_prompt)

        stream_result = ""

        def stream_data():
            for chunk in llm_agent.stream([{'role': 'user', 'text': rag_prompt}]):
                global stream_result
                stream_result += chunk.content
                yield chunk

        st.write_stream(stream_data)
        st.dataframe(query_df)

    # add message to history
    st.session_state.messages.append({"role": "ai", "text": stream_result})
