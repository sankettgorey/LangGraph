from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated, List

import operator
from dotenv import load_dotenv

load_dotenv('./langgraph_tutorial/.env')

search = TavilySearch(search_results = 3)

@tool
def addition(a: int, b: int):
    """takes two numbers and returns their addition"""
    return a + b


def division(a: int, b: int):
    """ takes two positive numbers and returns their division"""
    if b > 0:
        return a / b
    return 'Please enter valid numbers'


tools = [addition, division, search]


llm = ChatOllama(model = 'qwen2.5:7b-instruct')

llm_with_tools = llm.bind_tools(tools)


generation_prompt = ChatPromptTemplate(
    [
        (
            'system',
            'You are an expert in answering the questions asked by user.'
            'Use the appropriate underlying tools when required and formulate a proper response for the asked questions one paragraph of 10-12 lines.'
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)

generation_chain = generation_prompt | llm_with_tools



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def generation_node(state: AgentState):
    response = generation_chain.invoke(state)

    return {'messages': [response]}


tools_node = ToolNode(tools)


graph = StateGraph(AgentState)
graph.add_node('generation_node', generation_node)
graph.add_node('tools', tools_node)

graph.add_edge(START, 'generation_node')
graph.add_conditional_edges('generation_node', tools_condition,)
graph.add_edge('tools', 'generation_node')

workflow = graph.compile()

initial_state = {'messages': [HumanMessage(content = 'whats the latest news on tariffs on India')]}

final_state = workflow.invoke(initial_state)

print(final_state)
print()
print(final_state['messages'][-1].content)