from langgraph.graph import StateGraph, END

from typing import TypedDict


class SimpleState(TypedDict):

    count: int


def increment(state: SimpleState):

    return {
        'count': state['count'] + 1
    }


def shall_continue(state: SimpleState):

    if state['count'] < 5:
        return 'continue'
    return 'stop'


graph = StateGraph(SimpleState)

graph.add_node('increment', increment)
graph.set_entry_point('increment')

graph.add_conditional_edges('increment', shall_continue, {'continue': 'increment', 'stop': END})


workflow = graph.compile()

initial_state = {'count': 0}

final_state = workflow.invoke(initial_state)

print(final_state)