from langgraph.graph import StateGraph, END

from typing import TypedDict, List


class SimpleState(TypedDict):

    count: int
    sum: int
    history: List[int]


def increment(state: SimpleState):

    new_count = state['count'] + 1
    history = state['history'] + [new_count]

    return {
        'count': new_count,
        'sum': state['sum'] + new_count,
        'history': history
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

initial_state = {
    'count': 0,
    'sum': 0,
    'history': []
}

final_state = workflow.invoke(initial_state)

print(final_state)