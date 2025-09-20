from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

load_dotenv()
"""
steps to setup in the firestore database
1. go to console.cloud.google.com
2. create a new project
3. enable firestore for the project
4. Retrieve your project ID from the project settings
5. set the environment variable for the project ID
6. Install the google-cloud cli and run gcloud auth application-default login(authenticate your account)
7. pip install langchain-google-firestore
8. Enable the firestore api from the api library(google cloud console) in the console
"""

projectId = "langchain-6f416"

chat_history = [] # instead of local store, use firestore to store the conversation

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
    
    
    

