from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama

from typing import TypedDict

model = ChatOllama(model='qwen2.5:7b-instruct')


class ChatbotState(TypedDict):

    topic: str
    outline: str
    blog_content: str
    eval: str



def generate_outline(state: ChatbotState):
    topic=state['topic']
    prompt = f"Generate a detailed outline for the blog for the given topic: {topic}"

    state['outline'] = model.invoke(prompt).content

    return state


def generate_blog(state: ChatbotState):

    outline = state['outline']
    topic = state['topic']

    prompt = f"Write a blog for the topic: {topic} based on the given outline: {outline}"

    blog_content = model.invoke(prompt)
    state['blog_content']=blog_content

    return state


def evaluation(state: ChatbotState):

    content = state['blog_content']

    prompt=f"Based on the depth, maturity of the subject, evaluate the following blog and rate it out of 10. \n{content}"

    score = model.invoke(prompt).content

    state['eval'] = score

    return state



graph = StateGraph(ChatbotState)

graph.add_node('generate_outline', generate_outline)
graph.add_node('generate_blog', generate_blog)
graph.add_node('evaluation', evaluation)

graph.add_edge(START, 'generate_outline')
graph.add_edge('generate_outline', 'generate_blog')
graph.add_edge('generate_blog', 'evaluation')
graph.add_edge('evaluation', END)


workflow=graph.compile()

