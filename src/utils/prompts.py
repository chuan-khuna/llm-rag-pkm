rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question, as json format:

```json
{context}
```

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""
