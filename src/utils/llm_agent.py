import google.generativeai as genai

import ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class LLMAgentBase:
    def __init__(self, model: str):
        self.llm = None

    def generate_message_history(self, messages):
        history = []
        for message in messages:
            if message['role'] == 'user':
                history.append(HumanMessage(message['text']))
            elif message['role'] == 'agent' or message['role'] == 'ai':
                history.append(AIMessage(message['text']))
            else:
                history.append(SystemMessage(message['text']))
        return history

    def answer(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(prompt)])
        return response.content

    def chat(self, messages: list[dict]) -> str:
        history = self.generate_message_history(messages)
        response = self.llm.invoke(history)
        return response.content


# class GeminiAgent:
#     def __init__(self, model: str, api_key: str):
#         genai.configure(api_key=api_key)
#         self.llm = genai.GenerativeModel(model)  # default model is gemini-1.5-flash

#     def answer(self, prompt: str) -> str:
#         response = self.llm.generate_content(prompt)
#         return response.text


class LangChainGeminiAgent(LLMAgentBase):
    def __init__(self, model: str, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)


class OllamaAgent(LLMAgentBase):
    def __init__(self, model: str, temperature: float):
        self.llm = ChatOllama(model=model, temperature=temperature)
