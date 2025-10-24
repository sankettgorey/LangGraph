from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableBranch, RunnableLambda

from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal


model = ChatOllama(model = 'mistral:7b-instruct')


class SentimentClass(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='sentiment of a review')

pydantic_parser = PydanticOutputParser(pydantic_object=SentimentClass)

string_parser = StrOutputParser()


classifier_prompt = PromptTemplate(
    template = 'classify the sentiment of the following text: {review}\n {format_instructions}',
    input_variables=['review'],
    partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)

positive_prompt = PromptTemplate(
    template = 'Write an appropriate feedback for the positive review: {review}',
    input_variables=['review']
)

negative_prompt = PromptTemplate(
    template = 'Write an appropriate feedback for the negative review: {review}',
    input_variables = ['review']
)


# conditional branching chain
conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_prompt | model | string_parser),
    (lambda x: x.sentiment == 'negative', negative_prompt | model | string_parser),
    RunnableLambda(lambda x: 'No appropriate response for the given review')
)


classifier_chain = classifier_prompt | model | pydantic_parser 

final_chain = classifier_chain | conditional_chain

output = final_chain.invoke(
    {
        'review': 'ambience is good but the food is bad.'
    }
)

print(output)