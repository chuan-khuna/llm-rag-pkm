rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question, as json format:

```json
{context}
```

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context.

Keep your answer concise and clear.

Answer:"""

# Also provide the reference to the context you used, by adding `ref_id` at the end of sentence. (Use IEEE citation style, e.g. [1])
