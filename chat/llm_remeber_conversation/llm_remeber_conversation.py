from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

load_dotenv()

chat_history = []

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature="0.6")
SystemMessage = SystemMessage(content="You are a helpful assistant.")

chat_history.append(SystemMessage)

while True:
    request = input("You: ")
    if request.lower() == 'exist':
        break
    
    chat_history.append(HumanMessage(content=request))
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: ", response.content)
    
    
    