from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages


from typing import TypedDict, List, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_tavily import TavilySearch

load_dotenv('./langgraph_tutorial/.env')


llm = ChatOllama(model = 'qwen2.5:7b-instruct')

search = TavilySearch(amx_results = 5)

@tool
def addition(a: int, b: int):
    """Add two integers and return the result."""
    return a + b


@tool
def division(a: int, b: int):
    """Divide a by b and return the result. b must be non-zero."""
    if b <= 0:
        return "Division by zero"
    return a / b


tools = [addition, division, search]

llm_with_tools = llm.bind_tools(tools)



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    count: int
    # feedback: str


generation_prompt = ChatPromptTemplate(
    [
        (
            'system',
            'You are an expert in answering the questions. Answer the asked question in 15-20 lines paragraph. Use the appropriate tools when required'
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)

generation_chain = generation_prompt | llm_with_tools



def generation_node(state: AgentState):
    response = generation_chain.invoke(state['messages'])

    return {'messages': [response]}



reflection_prompt = ChatPromptTemplate(
    [
        (
            'system',
            'You are a critique who gives feedback for the answer. Give 3-4 line feedback based on the completeness of the answer for the asked qustion. Be specific about what is missing.'
        ),
        MessagesPlaceholder('messages')
    ]
)
reflection_chain = reflection_prompt | llm

def reflection_node(state: AgentState):
    # critique = llm.invoke(
    #     [HumanMessage('Give 2-3 line feedback based on the depth of the answer for the asked question')] + state['messages']
    # )
    critique = reflection_chain.invoke(state['messages'])

    return {
        'messages': [HumanMessage(critique.content)],
        # 'messages': state['feedback'],
        'count': state['count'] + 1
    }


modification_prompt = ChatPromptTemplate(
    [
        (
            'system',
            'Improve the previous answer based on the given fedback.'
        ),
        MessagesPlaceholder('messages')
    ]
)
modification_chain = modification_prompt | llm_with_tools

def modification_node(state: AgentState):
    # improved = llm.invoke(
    #     [HumanMessage('Improve the previous answer based on the reflection feedback message')] + state['messages']
    # )

    improved = modification_chain.invoke(
        {
            'messages': state['messages'],
            # 'feedback': state['feedback'],
        }
    )

    return {'messages': [improved]}


def generation_router(state: AgentState):
    last = state['messages'][-1]

    if hasattr(last, 'tool_calls') and last.tool_calls:
        return 'tools'
    return 'reflection'


def modification_router(state: AgentState):
    last = state['messages']

    if hasattr(last, 'tool_calls') and last.tool_calls:
        return 'tools'
    
    if state['count'] >= 2:
        return END
    return 'reflection'


tools_node = ToolNode(tools)

graph = StateGraph(AgentState)

graph.add_node('generation_node', generation_node)
graph.add_node('reflection', reflection_node)
graph.add_node('modification', modification_node)
graph.add_node('tools', tools_node)


graph.add_edge(START, 'generation_node')
graph.add_conditional_edges('generation_node', generation_router,
                            {
                                'tools': 'tools',
                                'reflection': 'reflection'
                            })

graph.add_edge('tools', 'generation_node')

graph.add_edge('reflection', 'modification')
graph.add_conditional_edges('modification', modification_router,
                            {
                                'tools': 'tools',
                                END: END,
                                'reflection': 'reflection'
                            }
                            )

workflow = graph.compile()

# ---------------------------
# RUN
# ---------------------------
result = workflow.invoke(
    {
        "messages": [HumanMessage(content="What's the latest update on tariffs on India?")],
        "count": 0
        # 'feedback': 'Null'
    }
)

print("\nFINAL ANSWER:\n")
# the last message in messages list should be an AIMessage/HumanMessage object
print(result["messages"][-1].content)

print(result)

