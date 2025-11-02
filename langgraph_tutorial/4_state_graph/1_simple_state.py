from langgraph.graph import StateGraph, MessageGraph, MessagesState, START, END

from langchain_ollama import ChatOllama

from typing import TypedDict


model = ChatOllama(model='qwen3:8b')


class SimpleState(TypedDict):

    count: int


def increment(state: SimpleState):

    print('In increment node')
    print(state['count'])
    
    state['count'] = state['count'] + 1
    return state



def shall_continue(state: SimpleState):

    count = state['count']

    if count > 5:
        return END
    
    return 'increment'


graph = StateGraph(SimpleState)

graph.add_node('increment', increment)
# graph.add_node('shall_continue', shall_continue)

graph.add_edge(START, 'increment')
graph.add_conditional_edges('increment', shall_continue)
# graph.add_edge(shall_continue, 'increment')


workflow = graph.compile()

# print(workflow.get_graph().draw_ascii())

initial_state = {'count': 0}

final_state = workflow.invoke(initial_state)

print(final_state)