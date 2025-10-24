from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser

from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableParallel

from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field

llm = ChatOllama(model = 'mistral:7b-instruct')


class Sentiment(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='sentiment/tone of the give text')
    # text: str

structured_model = llm.with_structured_output(Sentiment)
# parser = PydanticOutputParser(pydantic_object=Sentiment)


classification_prompt = PromptTemplate(
    template = 'Classify the given text into positive or negative: {text}',
    input_variables=['text'],
    # partial_variables={'format_instructions': parser.get_format_instructions()}
)

positive_prompt =PromptTemplate(
    template = "Write an appropriate positive feedback for the given text: {text}.",
    input_variables=['text']
)

negative_prompt = PromptTemplate(
    template = "Write an appropriate negative deedback for the given text: {text}",
    input_variables=['text']
)

llm_with_structure = llm.with_structured_output(schema=Sentiment)

classification_chain = classification_prompt | llm_with_structure

# print(classification_chain.invoke({'text': 'this is bad'}))

str_parser = StrOutputParser()


conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_prompt | llm | str_parser),
    (lambda x: x.sentiment == 'negative', negative_prompt | llm | str_parser),
    RunnableLambda(lambda x: 'No appropriate class found for the given text')
)

try:

    chain = classification_chain | conditional_chain

    output = chain.invoke(
        {
            'text': 'I didt like your product at all.'
        }
    )

    print(output)

except Exception as e:
    print(str(e))