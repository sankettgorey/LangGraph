from loguru import logger

from langgraph.graph import START, END, StateGraph

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.agents import AgentAction, AgentFinish

from typing import TypedDict, List, Annotated, Union
import operator
from loguru import logger

# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_react_agent


# from langchain_community.tools import tavily_search
from langchain_tavily import TavilySearch
from langchain import hub
from langchain.tools import tool
from langchain_ollama import ChatOllama

import datetime

from dotenv import load_dotenv
load_dotenv()

@tool
def get_system_time(format: str = '%Y-%m%d %h:%m:%s'):
    """gives date and time of the system in the given format"""

    return datetime.datetime.now().strftime(format)


search = TavilySearch(search_depth='basic')

llm = ChatOllama(model = 'qwen2.5:7b-instruct')
# llm = ChatOllama(model = 'qwen3:8b')
# llm = ChatOllama(model = 'qwen2.5:14b')
# llm = ChatOllama(model = 'mistral:7b-instruct')


prompt = hub.pull("hwchase17/react")


tools = [get_system_time, search]

# print('line 31')
react_agent = create_react_agent(
    llm = llm,
    prompt = prompt,
    tools = tools
)

# response = react_agent.invoke({
#     "input": "What year is it?",
#     "intermediate_steps": []
# })
# print(response)





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

