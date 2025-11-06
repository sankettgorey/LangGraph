from loguru import logger

from langgraph.graph import START, END, StateGraph

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.agents import AgentAction, AgentFinish

from typing import TypedDict, List, Annotated, Union
import operator
from loguru import logger

from react_runnable import react_agent, tools




class AgentState(TypedDict):

    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]


def reason_node(state: AgentState):
    agent_outcome = react_agent.invoke(state)
    logger.info('IN REASON NODE')
    logger.info(agent_outcome)

    return {'agent_outcome': agent_outcome}


def act_node(state: AgentState):
    logger.info('IN ACT NODE')
    agent_action = state['agent_outcome']
    logger.info(agent_action)

    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    tool_function = None

    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break
    
    if tool_function is None:
        output = f'{tool_name} not found'

    else:
        output = tool_function.invoke(tool_input)

    return {
        'intermediate_steps': [(agent_action, str(output))]
    }


def shall_continue(state: AgentState):
    logger.info('IN SHALL CONTINUE NODE')
    logger.info(state['agent_outcome'])
    

    if isinstance(state['agent_outcome'], AgentFinish):
        return END
    return 'act'



graph = StateGraph(AgentState)
graph.add_node('reason', reason_node)
graph.add_node('act', act_node)

graph.add_edge(START, 'reason')
# graph.set_entry_point('reason')

# graph.add_edge('reason', 'act')
graph.add_conditional_edges('reason', shall_continue)
graph.add_edge('act', 'reason')



workflow = graph.compile()

initial_state = {
    # 'input': 'when did spaceX launched its last rocket?',
    "input": "is there a holiday to NSE on 5th Nov 2025?",
    'agent_outcome': None,
    'intermediate_steps': []
}


try:
    final_state = workflow.invoke(initial_state)
    logger.info(final_state)

    print()
    print(final_state['agent_outcome'].return_values['output'], 'final result')

except Exception as e:
    print(e)

