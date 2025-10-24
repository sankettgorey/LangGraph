import asyncio
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MultiplySchema(BaseModel):
    a: float = Field(..., description="first number to multiply")
    b: float = Field(..., description="second number to multiply")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "multiplies two numbers"
    args_schema: Type[BaseModel] = MultiplySchema

    # required by BaseTool (abstract)
    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution")

    # actual async logic
    async def _arun(self, a: float, b: float) -> float:
        await asyncio.sleep(0)  # simulate async I/O
        return a * b


async def main():
    tool = MultiplyTool()
    result = await tool.ainvoke({"a": 10, "b": 5})
    print("✅ Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
