from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


schema = [
    ResponseSchema(name = 'fact1', description='deacription of fact1'),
    ResponseSchema(name = 'fact2', description='deacription of fact2'),
    ResponseSchema(name = 'fact3', description='deacription of fact3'),
]

json_parser = StructuredOutputParser.from_response_schemas(response_schemas=schema)


try:
    model = ChatOllama(model = 'mistral:7b-instruct')

    str_parser = StrOutputParser()

    prompt1 = PromptTemplate(
        template = 'Give me detailed report o the topic: {topic}',
        input_variables=['topic'],
    )

    prompt2 = PromptTemplate(
        template = 'summarize the text into 5 main points: {text}\n {format_instructions}',
        input_variables=['text'],
        partial_variables={'format_instructions': json_parser.get_format_instructions()}
    )

    chain = prompt1 | model | str_parser | prompt2 | model | json_parser

    output = chain.invoke(
        {'topic': 'black holes'}
    )

    print(output)
    

except Exception as e:
    print(str(e))

