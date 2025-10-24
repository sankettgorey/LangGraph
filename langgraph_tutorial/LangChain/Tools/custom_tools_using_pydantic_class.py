from langchain_community.tools import StructuredTool

from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: float = Field(..., description='first numner to multiply')
    b: float = Field(..., description='second number to multiply')

def multiply(a: float, b: float):
    return a * b


multiplyTool = StructuredTool.from_function(
    func=multiply,
    description='multiplies two numbers',
    args_schema=MultiplyInput
)


output = multiplyTool.invoke(
    {
        'a': 10,
        'b': 20
    }
)

print(output)