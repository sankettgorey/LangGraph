# we will create bmi calculator workflow in this program without LLM

'''
we will give weight and height as an input from one node and in the second node,
we will calculate BMI.

there will be three variables in the state:
weight: float
height: float
bmi: actual bmi value to calculate.
'''

from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


model=ChatOllama(model='qwn2.5:7b')



class BMIState(TypedDict):      # creating state of variables
    
    weight_kg: float
    height_kg: float
    bmi: float



'''
there are five main steps in executing the graph
1. create graph
2. create the nodes
3. create edges
4. compile the graph
5. execute the graph
'''


# creating node function
def calculate_bmi(state: BMIState):

    state['bmi']=state['weight_kg']/(state['height_kg'] ** 2)

    return state['bmi']


# creating graph
graph = StateGraph(state_schema=BMIState)

graph.add_node('calculate_bmi', calculate_bmi)

graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', END)

workflow = graph.compile()

print(Image(workflow.get_graph().draw_mermaid()))