from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Tweet Generation Prompt
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant that writes engaging, high-quality tweets.
            Your task:
            1. Generate a short, catchy tweet (under 280 characters) based on the topic provided.
            2. Wait for user feedback.
            3. When feedback is given, improve the tweet while keeping the same core message.

            Guidelines:
            - Use a natural, conversational tone.
            - Avoid hashtags unless they add real value.
            - Keep it easy to read and emotionally engaging.
            - Make each revision clearly better based on feedback.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Critique/Reflection Prompt
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a writing coach helping improve tweet quality.

            Given a tweet, evaluate it based on:
            1. Clarity — Is the message easy to understand?
            2. Engagement — Does it make people want to respond, share, or think?
            3. Emotional Impact — Does it trigger curiosity, motivation, or humor?
            4. Brevity — Is it concise but complete?

            Then, give a short critique and 1–2 actionable improvement suggestions.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Create prompt chains
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
