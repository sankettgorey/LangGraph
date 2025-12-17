from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages


from typing import TypedDict, List, Annotated, Literal, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_tavily import TavilySearch

load_dotenv('./langgraph_tutorial/.env')


llm = ChatOllama(model = 'qwen2.5:7b-instruct')


class SentimentSchema(BaseModel):
    
    sentiment: Literal['positive', 'negative'] = Field(description='sentiment of the given review')


class DiagnosisSchema(BaseModel):
    issue_type: str = Field(description='category of the problem faced by the customer')
    tone: Literal['angry', 'frustrated', 'disappointed', 'calm'] = Field(description='emotional tone of the review')
    urgency: Literal['high', 'medium', 'low'] = Field(description='how urgent the issue appears to be')



structured_model1 = llm.with_structured_output(SentimentSchema)
structured_model2 = llm.with_structured_output(DiagnosisSchema)


class ReviewState(TypedDict):
    review: str
    sentiment: Literal['pisitive', 'negative']
    diagnosis: Dict
    response: str



def find_sentiment(state: ReviewState):
    prompt = f"Find the sentiment of the following review: \n\nReview: {state['review']}"
    sentiment = structured_model1.invoke(prompt)

    print(f'Sentiment: {sentiment.sentiment}')

    return {'sentiment': sentiment.sentiment}



def check_sentiment(state: ReviewState):
    if state['sentiment'] == 'positive':
        return 'positive_response'

    return 'run_diagnosis'


def run_diagnosis(state: ReviewState):

    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
"""
    response = structured_model2.invoke(prompt)

    return {'diagnosis': response.model_dump()}

def negative_response(state: ReviewState):

    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = llm.invoke(prompt).content

    return {'response': response}


def positive_response(state: ReviewState):

    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n
Also, kindly ask the user to leave feedback on our website."""
    
    response = llm.invoke(prompt).content

    return {'response': response}



graph = StateGraph(ReviewState)

graph.add_node('find_sentiment', find_sentiment)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)
graph.add_node('positive_response', positive_response)

graph.set_entry_point('find_sentiment')
graph.add_edge('run_diagnosis', 'negative_response')

graph.add_conditional_edges('find_sentiment', check_sentiment,
                            {
                                'run_diagnosis': 'run_diagnosis',
                                'positive_response': 'positive_response'
                            })


workflow = graph.compile()

intial_state ={
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality.",
    'sentiment': 'None',
    'diagnosis': {},
    'response': 'None'
}

output = workflow.invoke(intial_state)
print(output)