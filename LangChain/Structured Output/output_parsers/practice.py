from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import PydanticOutputParser

from pydantic import Field, BaseModel
from typing import Literal, Optional


class Person(BaseModel):

    name: str = Field(description='name of a person')
    age: int = Field(description='age of a person')
    gender: Optional[Literal['male', 'female']] = Field(description='gender of a person')


model = ChatOllama(model = 'mistral:7b-instruct')


parser = PydanticOutputParser(pydantic_object=Person)


prompt = PromptTemplate(
    template = 'give me name, age, gender of a {person} person. \n{format_instuctions}',
    input_variables= ['person'],
    partial_variables={'format_instuctions': parser.get_format_instructions()}
)


chain = prompt | model | parser


try:
    output = chain.invoke({'person': 'indian'})
    print(output)

except Exception as e:
    print(e)