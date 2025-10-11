from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

generate_joke_prompt = ChatPromptTemplate.from_messages((
    "You are a professional joke teller. Generate one short joke per request."
))

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

generate_chain = generate_joke_prompt | llm