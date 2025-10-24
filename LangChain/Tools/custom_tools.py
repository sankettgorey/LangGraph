from langchain.tools import StructuredTool, tool

def add(a: int, b: int) -> int:
    """accepts two integers and add them"""
    return a + b

# Wrap using StructuredTool
add_tool = StructuredTool.from_function(add)

# Now you can call it with dict input
output = add_tool.invoke({"a": 10, "b": 5})
print(output)  # 15



@tool
def multiply(a: float, b: float):
    '''multiplies two numbers'''
    return f'output is: {a * b}'


output = multiply.invoke(input={
    'a': 10,
    'b': 23
})

print(output)