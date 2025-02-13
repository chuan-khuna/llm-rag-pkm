from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def generate_message_history(messages):
    history = []
    for message in messages:
        if message['role'] == 'user':
            history.append(HumanMessage(message['text']))
        elif message['role'] == 'agent' or message['role'] == 'ai':
            history.append(AIMessage(message['text']))
        else:
            history.append(SystemMessage(message['text']))
    return history
