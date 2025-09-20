from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate 

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

template = "Write a {tone} email to {company} expressing interest in a job opening for the position of {position}, highlighting my skills in {skills}. Keep it under {line} words."

prompt_template = PromptTemplate.from_template(template)

# print(prompt_template.input_variables, prompt_template)

prompt_values = {
    "tone": "professional",
    "company": "ABC Corp",
    "position": "Software Engineer",
    "skills": "Python, Machine Learning",
    "line": 100
}

prompt = prompt_template.invoke(prompt_values)
response = llm.invoke(prompt)
print(response.content)

# we can have a bit of control in both the the system and human prompts(we can do that in over code)