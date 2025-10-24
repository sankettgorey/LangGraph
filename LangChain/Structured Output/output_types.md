There are three  types of structured outouts
- TypedDict
- json schema
- Pydantic


# TypedDict
- This is the way of creating dictionary which follows specific structure.
- This tells the program which keys have which data types of values.
- We explicitely specify value's datatypes which declaring class of TypedDict.
- This does not enforce any data types though. We can give another type's data type without getting any error

# Pydantic
- This is the way of data validation in python.
- Pydantic validates the data types and data classes such that if given variable doesn't follow the specified format, it gives error.


# Output Parsers
- There are three types of output parsers we use mainly in Langchain:
    - StrOutputParser
    - JsonOutputParser
    - StructuredOutputParser

- StrOutputParser:
    - This is used to get the string format output from model. We form a chain and pass output of the model to parser to get final string output.

    - output = template1 | model | parser
    - This will give string outout directly

- JsonOutputParser:
    - This is used to get json format output from model.
    - We have to pass `partial_variables` in the prompt template to tell the model that we want output in the json format.
    - The main problem of this parser is that we can't enforce output schema to the model with this parser which is overcame by structured output parser.

- Structured Output Parser:
    - This is the ouptut parser which helps extract structued JSON data from LLM responses based on predefined field schemas.