from typing import Annotated, TypedDict, Optional, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_tavily import TavilySearch

from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph

from dotenv import load_dotenv
load_dotenv('./langgraph_tutorial/.env')


# ===============================
# LLM
# ===============================
generation_llm = ChatOllama(model = 'qwen3:8b')
feedback_llm = ChatOllama(model = 'qwen3:8b')

# ===============================
# TOOLS
# ===============================

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

search = TavilySearch(max_results = 3)

tools = [multiply, search]

tool_node = ToolNode(tools = tools)

# ===============================
# PROMPTS
# ===============================
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are an expert in answering the questions asked by user. Answer each question in detail. If there is a feedback given, modify the answer as per feedback.'
        )
        ,
        MessagesPlaceholder(variable_name = 'messages')
    ]
)
generation_chain = generation_prompt | generation_llm.bind_tools(tools)


evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'YOu are a strict reviewer. '
            "If the answer is correct and complete, reply with 'APPROVE'. "
            'Otherwise, explain clearely what is wrong and how to fic it. '
        ),
        MessagesPlaceholder(variable_name = 'messages')
    ]
)

evaluation_chain = evaluation_prompt | feedback_llm


# ===============================
# STATE
# ===============================

class AgentState(TypedDict):

    messages: Annotated[List[BaseMessage], add_messages]
    count: int
    final_answer: Optional[str]

# ===============================
# NODES
# ===============================
def generation_node(state: AgentState):
    output = generation_chain.invoke(state['messages'])

    final_answer = state['final_answer']

    if isinstance(output, AIMessage) and not output.tool_calls:
        final_answer = output.content
    return {
        'messages': [output],
        'final_answer': final_answer
    }

def evaluation_node(state: AgentState):
    evaluation = evaluation_chain.invoke(state['messages'])

    count = state.get('count', 0) + 1

    feedback = SystemMessage(
        content = f"Reviewer Feedback:\n{evaluation.content}"
    )

    return {
        'messages': [feedback],
        'count': count
    }

def generation_condition(state: AgentState):
    last = state['messages'][-1]
    print(last)
    if state['count'] >= 1:
        return END
    
    if last.content == 'APPROVE':
        return END
    
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return 'tools'
    
    return 'evaluation'


# ===============================
# GRAPH
# ===============================
graph = StateGraph(AgentState)

graph.add_node('generation', generation_node)
graph.add_node('evaluation', evaluation_node)
graph.add_node('tools', tool_node)

graph.set_entry_point('generation')
graph.add_conditional_edges('generation', generation_condition,
                            {
                                'evaluation': 'evaluation',
                                END: END,
                                'tools': 'tools'
                            })

graph.add_edge('tools', 'generation')

graph.add_edge('evaluation', 'generation')

workflow = graph.compile()

# ===============================
# RUN
# ===============================

initial_state = {'messages': [HumanMessage(content = "What's the latest news on Epstein files?")], 'count': 0, 'final_answer': 'None'}

result = workflow.invoke(initial_state)
print(result)
print()
print('FINAL ANSWER\n')
print(result['final_answer'])