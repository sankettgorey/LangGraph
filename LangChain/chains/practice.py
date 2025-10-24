from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from langchain.schema.runnable import RunnableLambda, RunnableBranch

from pydantic import BaseModel, Field

from typing import Literal, TypedDict, Annotated


model = ChatOllama(model = 'mistral:7b-instruct')


class Sentiment(TypedDict):
    sentiment: Literal['positive', 'negative']
    review: str

str_parser = StrOutputParser()

structured_model = model.with_structured_output(schema = Sentiment)


sentiment_template = PromptTemplate(
    template='tell me whether given review is positive or negative: {review}',
    input_variables=['review'],
)


positive_template = PromptTemplate(
    template = 'write a feedback for a given positive review: {review}',
    input_variables=['review']
)

negative_template = PromptTemplate(
    template = 'write an appropriate feedback for the given negative review: {review}',
    input_variables=['review']
)


classification_chain = sentiment_template | structured_model


conditional_chain = RunnableBranch(
    (lambda x: x['sentiment'] == 'positive', positive_template | model | str_parser),
    (lambda x: x['sentiment'] == 'negative', negative_template | model | str_parser),
    RunnableLambda(lambda x: 'No specific class found for given review')
)


try:

    chain = classification_chain | conditional_chain

    output = chain.invoke(
    {
        'review': 'this is bad phone'
    }
)

    print(output)

except Exception as e:
    print(str(e))