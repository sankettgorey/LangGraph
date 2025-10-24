from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_core.output_parsers import JsonOutputParser

# model = ChatOllama(model = 'mistral:7b-instruct')
model = ChatOllama(model = 'llama3.2:3b')


try:
    # defining schema in json format
    schema = [
        ResponseSchema(name = 'fact1', description = 'fact1 about the topic'),
        ResponseSchema(name = 'fact2', description = 'fact2 about the topic'),
        ResponseSchema(name = 'fact3', description = 'fact3 about the topic')
    ]


    parser = JsonOutputParser()

    template = PromptTemplate(
        template = 'give me name, age, city of any indian person\n {format_instructions}',
        input_variables=['topic'],
        partial_variables={'format_instructions': parser.get_format_instructions()}
    )


    chain = template | model | parser

    print(
        chain.invoke({'topic': 'supply chain'})
    )

except Exception as e:
    print(e)