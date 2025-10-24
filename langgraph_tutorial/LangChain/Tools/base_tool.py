from langchain.tools import BaseTool
from typing import Any, Type
from pydantic import BaseModel, Field


class MultiplySchema(BaseModel):
    a: float = Field(..., description="first number to multiply")
    b: float = Field(..., description="second number to multiply")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "multiplies two numbers"

    args_schema: Type[BaseModel] = MultiplySchema

    def _run(self, a: float, b: float) -> float:
        # async version only
        
        return a * b


tool = MultiplyTool()

output = tool.invoke({"a": 10, "b": 5})
print(tool.args)


