# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# llm = ChatOpenAI(model="o4-mini", temperature=0.5)

# result = llm.invoke("Hello, how are you?")

# print("llm result", result)

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

# type of message in langchain
# 1. systemMessage, HumanMessage and AiMessage


def ChatWithGenAi ():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
        temperature=0.5)
    result = llm.invoke("Hello, How are you doing? and What is the latest ai model in google?")
    return result.content


print("This is response form gen ai", ChatWithGenAi())