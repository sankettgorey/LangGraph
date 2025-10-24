from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage

# 1. Load model
model = ChatOllama(model="llama3.2:3b")

# 2. Define prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support executive"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")   # ✅ fixed placeholder
])

# 3. Chat history
chat_history = []

print("Customer Support Chatbot (type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message
    chat_history.append(HumanMessage(content=user_input))

    # Build prompt with history
    messages = chat_template.format_messages(
        chat_history=chat_history, 
        user_input=user_input
    )

    # Model response
    response = model.invoke(messages)
    print("Assistant:", response.content)

    # Add AI response to history
    chat_history.append(AIMessage(content=response.content))
