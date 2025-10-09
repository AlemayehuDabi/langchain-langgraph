# from langchain_tavily import TavilySearch
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Original Tavily search
tavily = TavilySearchResults(search_depth="basic", max_lenght=5)

# Wrapper to make it single-input
search_tool = Tool(
    name="tavily_search",
    func=lambda query: tavily.run(query),
    description="Use this tool to search the web for a single query"
)

# Initialize agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    # handle_parsing_errors=True
)

# Test it
# response = 
agent.invoke("Give me a funny tweet about today's weather in Washington DC")

# print("\n🧠 Agent Response:\n", response["output"])
