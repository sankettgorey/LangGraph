from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from typing import List, TypedDict
from chains import actor_chain, revisor_chain
from tools_execution import execute_tools, ReflexionState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv('../.env')

llm = ChatOpenAI(model='gpt-5-nano')

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

    iteration = state["iteration"] + 1
    print(f"Actor producing iteration {iteration}")
    # print(result)

    return {
        'messages': state['messages']
        + [
            AIMessage(content=result.tool_calls[0]['args']['answer']),
            AIMessage(content=f"critique: {result.tool_calls[0]['args']['reflection']}")
        ] + [AIMessage(content=f"tool_calls={[str(result.tool_calls[0])]}")],
        "iteration": iteration,
    }



def reflection(state: ReflexionState):
    iteration = state["iteration"] + 1
    print('Inside reflection chain')
    print('=' * 20)
    result = revisor_chain.invoke(state['messages'])

    # print(result)

    # ✅ Important: pass iteration forward!
    return {
        "messages": state["messages"]
        + [AIMessage(content=result.tool_calls[0]['args']['answer'])],
        "iteration": iteration,
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
        'iteration': 0
    }
    final_state = workflow.invoke(initial_state)
    print(final_state)
    print('=' * 20)
    print(final_state['messages'][-1].content)
except Exception as e:
    print("❌ Error:", str(e))




x = {'messages': [HumanMessage(content='write a report on the future of ai', additional_kwargs={}, response_metadata={}), 
                  AIMessage(content='The future of AI is poised to revolutionize industries, society, and science. Advancements in machine learning, natural language processing, and computer vision will drive automation, personalized services, and scientific discovery. By 2030, AI could achieve human-level performance in specific tasks, enabling breakthroughs in healthcare (e.g., personalized medicine), climate modeling, and autonomous systems. However, challenges like data privacy, algorithmic bias, and ethical deployment will require robust governance frameworks. Quantum computing may unlock new capabilities, while brain-computer interfaces could merge human cognition with AI. Yet, risks such as job displacement, surveillance overreach, and existential threats demand proactive regulation. The path forward hinges on interdisciplinary collaboration between technologists, policymakers, and ethicists to ensure AI aligns with human values. While optimism is warranted, overhyped timelines and underestimating societal resistance could derail progress. The future will likely involve a hybrid model where AI augments human potential rather than replaces it, though the exact trajectory remains uncertain.', additional_kwargs={}, response_metadata={}), 
                  AIMessage(content="critique: {'issues': ['Overly optimistic timeline projections lack citation', 'Insufficient discussion of AI alignment challenges', 'Underemphasizes geopolitical competition risks'], 'severity': 'high'}", additional_kwargs={}, response_metadata={}), 
                  AIMessage(content='tool_calls=["{\'name\': \'AnswerQuestion\', \'args\': {\'answer\': \'The future of AI is poised to revolutionize industries, society, and science. Advancements in machine learning, natural language processing, and computer vision will drive automation, personalized services, and scientific discovery. By 2030, AI could achieve human-level performance in specific tasks, enabling breakthroughs in healthcare (e.g., personalized medicine), climate modeling, and autonomous systems. However, challenges like data privacy, algorithmic bias, and ethical deployment will require robust governance frameworks. Quantum computing may unlock new capabilities, while brain-computer interfaces could merge human cognition with AI. Yet, risks such as job displacement, surveillance overreach, and existential threats demand proactive regulation. The path forward hinges on interdisciplinary collaboration between technologists, policymakers, and ethicists to ensure AI aligns with human values. While optimism is warranted, overhyped timelines and underestimating societal resistance could derail progress. The future will likely involve a hybrid model where AI augments human potential rather than replaces it, though the exact trajectory remains uncertain.\', \'reflection\': {\'issues\': [\'Overly optimistic timeline projections lack citation\', \'Insufficient discussion of AI alignment challenges\', \'Underemphasizes geopolitical competition risks\'], \'severity\': \'high\'}, \'search_queries\': [\'recent AI ethics framework developments\', \'quantum machine learning breakthroughs 2023-2024\', \'AI governance policy trends in EU/US\']}, \'id\': \'dd8c4085-17e1-4b38-a747-ca53db98383f\', \'type\': \'tool_call\'}"]', additional_kwargs={}, response_metadata={}), 
                  AIMessage(content="The future of AI will hinge on balancing innovation with governance. While advancements in machine learning, NLP, and computer vision will drive automation and scientific discovery, timeline projections for human-level AI by 2030 lack consensus. A 2023 MIT report notes that while narrow AI advances are likely, general AI remains speculative [1]. Critical challenges include AI alignment—ensuring systems prioritize human values—and mitigating risks like reward misalignment and value learning failures [2]. Geopolitical competition, particularly between the U.S., China, and EU, will shape AI development, with nations investing heavily in quantum computing and brain-computer interfaces [3]. Ethical frameworks, such as the EU's AI Act, aim to regulate high-risk applications, but global coordination remains fragmented. While AI could transform healthcare and climate science, risks like job displacement and surveillance require proactive policy. The path forward demands interdisciplinary collaboration to align technical progress with societal needs, though uncertainties about adoption rates and societal resistance persist.", additional_kwargs={}, response_metadata={})], 'iteration': 2}
