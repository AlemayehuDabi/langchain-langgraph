from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import yaml

llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Fast and capable
    temperature=0.7,
    api_key="your_groq_api_key_here"
)

