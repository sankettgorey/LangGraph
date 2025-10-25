from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from typing import List, TypedDict
from chains import actor_chain, revisor_chain
from tools_execution import execute_tools, ReflexionState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from loguru import logger

load_dotenv('../.env')



MAX_ITERATIONS = 2


def event_loop(state: ReflexionState) -> str:
    iteration = state['iteration']
    print(iteration)
    print(f"Iteration check → {iteration}")
    if iteration >= MAX_ITERATIONS:
        print("✅ Reached max iterations. Stopping.")
        return END
    return "execute_tools"


def actor(state: ReflexionState):
    result = actor_chain.invoke(input={'messages': state['messages']})
    print(state['references'])

    iteration = state["iteration"] + 1
    print(f"Actor producing iteration {iteration}")

    ai_message = [AIMessage(content = f"{result.tool_calls[0]['args']['answer']} \n {result.tool_calls[0]['args']['reflection']}")]
    
    result_to_update = state['messages'] + ai_message


    return {
        'messages': result_to_update,
        "iteration": iteration,
        "tool_calls": [result.tool_calls[0]],
        "references": state['references']
    }



def reflection(state: ReflexionState):
    iteration = state["iteration"] + 1
    print('Inside reflection chain')
    # print(state)
    print('=' * 20)
    result = revisor_chain.invoke(state['messages'])

    print(result.tool_calls[0]['args']['references'])

    # ✅ Important: pass iteration forward!
    return {
        "messages": state["messages"]
        + [AIMessage(content=result.tool_calls[0]['args']['answer'])],
        "iteration": iteration,
        "tool_calls": [result.tool_calls[0]],
        "references": result.tool_calls[0]['args']['references']
    }


graph = StateGraph(ReflexionState)
graph.add_node('draft', actor)
graph.add_node('execute_tools', execute_tools)
graph.add_node('revisor', reflection)

graph.add_edge(START, 'draft')
graph.add_edge('draft', 'execute_tools')
graph.add_edge('execute_tools', 'revisor')

graph.add_conditional_edges('revisor', event_loop)

workflow = graph.compile()

try:
    initial_state = {
        'messages': [HumanMessage(content='write a report on the future of ai')],
        'iteration': 0,
        "tool_calls": None,
        "references": None
    }
    final_state = workflow.invoke(initial_state)
    print(final_state)
    print('=' * 20)

    print(f"{final_state['messages'][-1].content} \n {final_state['references']}")
    print(*final_state['references'])
except Exception as e:
    print("❌ Error:", str(e))
    logger.error(str(e))




