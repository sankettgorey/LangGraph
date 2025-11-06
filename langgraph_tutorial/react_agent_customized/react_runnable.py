from loguru import logger

from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.agents import AgentAction, AgentFinish

from typing import TypedDict, List, Annotated, Union
import operator

from langchain_tavily import TavilySearch
from langchain_community.llms.ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent, AgentType
from langchain.tools import tool
import datetime

from dotenv import load_dotenv
load_dotenv()


@tool
def get_system_time(format: str = '%Y-%m-%d %H:%M:%S'):
    """gives date and time of the system in the given format"""
    return datetime.datetime.now().strftime(format)

search = TavilySearch(search_depth='basic')
llm = ChatOllama(model='qwen2.5:7b-instruct')
prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Use the tools you have when needed.\n\n{input}"
)
tools = [get_system_time, search]

# Use new agent creation API in v1
agent = create_agent(
    # agent_type=AgentType.OPENAI_FUNCTIONS,  # Or use AgentType.REACT_FUNCTIONS if available, see docs.
    agent_type = AgentType.OPENAI_FUNCTIONS
    llm=llm,
    prompt=prompt,
    tools=tools
)

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

def reason_node(state: AgentState):
    agent_outcome = agent.invoke({
        "input": state["input"],
        "intermediate_steps": state["intermediate_steps"]
    })
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
graph.add_conditional_edges('reason', shall_continue)
graph.add_edge('act', 'reason')
workflow = graph.compile()

initial_state = {
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
