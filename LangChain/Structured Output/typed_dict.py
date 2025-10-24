from typing import TypedDict



class Person(TypedDict):
    name: str
    age: int

new_person: Person = {
    'name': 'sanket',
    'age': 35
}


from typing import TypedDict, Annotated, Literal, Optional

from langchain_ollama import ChatOllama


model = ChatOllama(model = 'llama3.2:3b')


class Review(TypedDict):
    summary: Annotated[str, "Summary of the review"]
    sentiment: Annotated[Literal['pos', 'neg'], 'sentiment of the review']
    theme: Annotated[Optional[str], 'what review is talking about on a broad level']

structured_model = model.with_structured_output(schema = Review)

review = '''
I had a great time working in the field of AI. But because it's evolving very rapidly, there are times when you feel challending to keep yourself upto date.
'''

result = structured_model.invoke(review)

print(result)

print()

print(result['summary'])
print()
print(result['sentiment'])
print()
print(result['theme'])