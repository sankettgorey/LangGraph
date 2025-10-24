from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field
from typing import Optional, Literal, List


llm = ChatOllama(model = 'llama3.2:3b')


class Review(BaseModel):
    
    summary: str = Field(description='summary of a review')
    setiment: Optional[Literal['pos', 'neg']] = Field(description='sentiment of a review')
    theme: List[str] = Field(description='what review is talking about')

model = llm.with_structured_output(schema = Review)


review = '''
I had a great time working in the field of AI. But because it's evolving very rapidly, there are times when you feel challending to keep yourself upto date.
'''

output = model.invoke(review)

print(output)

print()


for key, value in output:
    print(f"{key}: {value}")