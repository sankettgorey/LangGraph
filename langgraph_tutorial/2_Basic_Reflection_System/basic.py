from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from chains import generation_chain, relfection_chain  # fixed typo here
from langchain_ollama import ChatOllama
from typing import TypedDict, List

llm = ChatOllama(model='qwen2.5:7b-instruct')

class Data(TypedDict):
    messages: List[BaseMessage]

def generation_node(state: Data):
    response = generation_chain.invoke({'messages': state['messages']})
    modified_state = {'messages': state['messages'] + [AIMessage(content=response.content)]}
    print("Generation Node State:", modified_state)
    return modified_state

def reflection_node(state: Data):
    response = relfection_chain.invoke({'messages': state['messages']})
    print('*' * 10)
    print("Reflection response:", response)
    new_state = {'messages': state['messages'] + [HumanMessage(content=response.content)]}
    return new_state

generate = 'GENERATE'
reflect = 'REFLECT'

def shall_continue(state: Data):
    if len(state['messages']) > 6:
        return END
    return reflect  # correctly returning next node key for conditional edge

graph = StateGraph(Data)
graph.add_node(generate, generation_node)
graph.add_node(reflect, reflection_node)
graph.set_entry_point(generate)

graph.add_conditional_edges(generate, shall_continue)  # controls whether to end or go to reflect
graph.add_edge(reflect, generate)  # after reflection, go back to generate

workflow = graph.compile()
