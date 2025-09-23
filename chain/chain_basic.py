from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import StrOutputParser

load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

prompt = [
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "Translate the following text from {input_language} to {output_language}: {text}")
]

prompt_template = ChatPromptTemplate.from_messages(prompt)

chain = prompt_template | llm | StrOutputParser()

response = chain.invoke({
    "input_language": "English",
    "output_language": "Amharic",
    "text": "Hello, how are you?"
})

print(response)# Basic chain example using ChatGoogleGenerativeAI and ChatPromptTemplate