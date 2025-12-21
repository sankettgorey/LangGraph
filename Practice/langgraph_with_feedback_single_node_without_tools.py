from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# ---------------------------
# ENV
# ---------------------------
load_dotenv("./langgraph_tutorial/.env")

# ---------------------------
# LLM
# ---------------------------
# llm = ChatOllama(model="qwen2.5:7b-instruct")
# llm = ChatOllama(model="qwen2.5:7b-instruct")
generator_llm = ChatOllama(model="qwen3:8b")
feedback_llm = ChatOllama(model="qwen3:8b")



class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    count : int
    final_answer: str


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
        'system',
        'You are an expert in answering the users question. Answer each question in detail as if you are explaining it to 6th grader student.'
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)
generation_chain = generation_prompt | generator_llm
def llm_node(state: AgentState):


    result = generation_chain.invoke(state['messages'])
    print(result)

    return {'messages': [result.content], 'final_answer': result.content}


feedback_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are an exper critique who constructively critisize the answer for the given question. '
            'Give feedback in 2-3 lines based on 1. Quality of the answer\n2. Depth of the answer\n3. Completeness of the answer '
        ),
        MessagesPlaceholder(variable_name = 'messages')
    ]
)
feedback_chain = feedback_prompt | feedback_llm
def feedback_node(state: AgentState):
    output = feedback_chain.invoke(state['messages'])
    # print(output)

    return {
        'messages': state['messages'] + [HumanMessage(content = output.content)],
        'count': state['count'] + 1
    }

def shall_continue(state:AgentState):
    if state['count'] > 1:
        return END
    return 'feedback'

graph = StateGraph(AgentState)
graph.add_node('llm', llm_node)
graph.add_node('feedback', feedback_node)

graph.add_edge(START, 'llm')
graph.add_conditional_edges('llm', shall_continue,
                            {
                                'feedback': 'feedback',
                                END: END
                            })
graph.add_edge('feedback', 'llm')

workflow = graph.compile()


while True:
    user = input('User: ')

    if user.lower() in ['quit', 'break', 'stop', 'exit']:
        break

    result = workflow.invoke(
        {
            'messages': [HumanMessage(content = user)],
            'count': 0
        }
    )

    print()
    print('FINAL ANSWER: ')
    print(result['final_answer'])
    print()
