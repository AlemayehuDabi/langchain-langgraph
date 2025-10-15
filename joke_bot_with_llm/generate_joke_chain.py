from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

generate_joke_prompt = ChatPromptTemplate.from_messages([(
    "You are a professional joke teller. Generate one short joke per request."
), MessagesPlaceholder(variable_name="messages")])

critic_joke_prompt = ChatPromptTemplate.from_messages([(
    "You are a professional joke critic. your job is to evaluate and critic the joke given"
), MessagesPlaceholder(variable_name="messages")])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

generate_chain = generate_joke_prompt | llm
critique_chain = critic_joke_prompt | llm