from chains import relfection_chain, generation_chain

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

from langgraph.graph import StateGraph, START, END

from typing import TypedDict, List, Annotated, Optional
import operator

llm=ChatOllama(model='qwen2.5:7b-instruct')


class TwitState(TypedDict):

    messages: List[BaseMessage]

    reflection_score:Optional[float]



def generate_tweet(state: TwitState):

    twit = generation_chain.invoke(
        input={
            'messages': state['messages']
        }
    )
    print(twit.content)
    print()
    return {'messages': state['messages'] + [AIMessage(content=twit.content)]}



def reflect_tweet(state: TwitState):

    reflection=relfection_chain.invoke(
        input={
            'messages': state['messages']
        }
    )
    new_message = [HumanMessage(content=reflection.reflection_message)]
    print(new_message)
    return {'messages': state['messages'] + new_message,
            'reflection_score': reflection.score}



def shall_continue(state: TwitState):
    print(len(state['messages']))
    if len(state['messages']) > 6 or state['reflection_score'] >= 7:
        return END
    
    return 'reflect'


graph=StateGraph(TwitState)


graph.add_node('generate', generate_tweet)
graph.add_node('reflect', reflect_tweet)


graph.add_edge(START, 'generate')
graph.add_conditional_edges('generate', shall_continue)
graph.add_edge('reflect', 'generate')


workflow=graph.compile()


initial_state = {'messages': [HumanMessage(content='generate a tweet on modi leading india')], 'reflection_score': 0}

output = workflow.invoke(initial_state)

print(output)
print()

print()
