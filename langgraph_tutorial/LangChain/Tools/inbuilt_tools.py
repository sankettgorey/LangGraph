from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

output = search.invoke('news on cricket')

print(output)