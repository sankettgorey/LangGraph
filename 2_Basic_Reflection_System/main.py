from chains import generation_chain
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from loguru import logger
from typing import TypedDict, Annotated, Optional, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from dotenv import load_dotenv

load_dotenv()

llm=ChatOllama(model='qwen2.5:7b-instruct')

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a twitter techie influencer assistant tasked with writing excellent twitter post."
            " Generate best twitter post possible for the user's request."
            " If user provides critique, respond with the revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)

class ReflectionSchema(BaseModel):
    reflection_message: str=Field(description='Two line critique about the generate tweet')
    score: float=Field(description='score for the generated tweet', ge=0, le=10)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system', 
            "You are a twitter techie influencer assistant tasked with writing excellent twitter post."
            " Generate best twitter post possible for the user's request."
            " If user provides critique, respond with the revised version of your previous attempts."
        ),
            MessagesPlaceholder(variable_name='messages')
    ]
)

relfection_chain = reflection_prompt | llm.with_structured_output(schema=ReflectionSchema)


'==================================================================='



class TwitterState(TypedDict):

    messages: List
    score: float


def generate_tweet(state: TwitterState):
    tweet = generation_chain.invoke(
        input={
            'messages': state['messages']
        }
    )

    return {'messages': state['messages'] + [AIMessage(content=tweet.content)]}


def reflect_tweet(state: TwitterState):
    reflect = relfection_chain.invoke(
        input={'messages': state['messages']})

    return {'messages': state['messages'] + [HumanMessage(content=reflect.reflection_message)], 'score': reflect.score}



def shall_continue(state: TwitterState):
    if len(state['messages']) > 6:
        return END
    return 'reflect'


graph=StateGraph(TwitterState)

graph.add_node('generate', generate_tweet)
graph.add_node('reflect', reflect_tweet)

graph.add_edge(START, 'generate')
graph.add_edge('reflect', 'generate')
graph.add_conditional_edges('generate', shall_continue)


workflow=graph.compile()

initial_state = {'messages': [HumanMessage(content='generate a tweet on dark reality of USA')]}

output = workflow.invoke(initial_state)

print(output)


