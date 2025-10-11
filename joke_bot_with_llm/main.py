from typing import Annotated, Literal
from pydantic import BaseModel
from pyjokes import get_joke
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from operator import add
from generate_joke_chain import generate_chain

class Joke(BaseModel):
    txt: str
    joke_category: str
    
class Joke_State(BaseModel):
    jokes: Annotated[list[Joke], add] = []
    joke_choice: Literal["n", "c", "q", "l", "r"] = "n"
    category: str = "neutral"
    language: str = "en"
    quit: bool = False
    
def show_menu(state: Joke_State) -> dict:
    print("============================================================")
    print(f"🎭 Menu | Category: {state.category} | Jokes: {len(state.jokes)}")
    print("--------------------------------------------------")
    print("Pick an option:")
    user_input = input("[n] 🎭 Next Joke  [c] 📂 Change Category [l] change Langugae [r] reset [q] 🚪 Quit").strip().lower()
    print("user_name: ", user_input)
    return { "joke_choice": user_input }

def fetch_joke(state: Joke_State) -> dict:
    prompt_text = (
        f"You are a professional joke teller. "
        f"Generate one short {state.category} joke in {state.language} language."
    )
    joke = generate_chain.invoke({"input": prompt_text})
    # print("this is right from the llm", joke.content)
    new_joke = Joke(txt=joke.content, joke_category=state.category)
    print(f"\n😂 {new_joke.txt}\n")
    # print(state.jokes)
    return { "jokes": [new_joke] }

def update_category(state: Joke_State) -> dict:
    categories = ["neutral", "chuck", "all"]
    selected_input = int(input("[0] - neutral, [1] - chuck, [2] - all").strip())
    return {"category": categories[selected_input]} 

def update_language(state: Joke_State) -> dict:
    categories = ["en", "de", "es", "it", "gl", "eu"]
    selected_input = int(input("[0] - english, [1] - German, [2] - Spanish, [3] - Italian, [4] - Galician, [5] - Basque"))
    return {"language": categories[selected_input]}

def reset_jokes(state: Joke_State) -> dict:
    state.jokes.clear()
    # print(state.jokes)
    return {}

def exit_bot(state: Joke_State) -> dict:
    print("🚪==========================================================🚪")
    print("GOODBYE!")
    print("============================================================")
    print("🎊==========================================================🎊")
    print("SESSION COMPLETE!")
    print("============================================================")
    print(f"📈 You enjoyed {len(state.jokes)} jokes during this session!")
    print(f"📂 Final category: {state.category}")
    print("🙏 Thanks for using the LangGraph Joke Bot!")
    print("============================================================")
    return { "quit": True}

def route_choice(state: Joke_State) -> str:
    if state.joke_choice == "n":
        return "fetch_joke"
    elif state.joke_choice == "c":
        return "update_category"
    elif state.joke_choice == "q":
        return "exit_bot"
    elif state.joke_choice == "l": 
        return "update_language"
    elif state.joke_choice == "r":
        return "reset_jokes"
    return "exit_bot"


def build_joke_graph() -> CompiledStateGraph:
    graph = StateGraph(Joke_State)
    
    graph.add_node("show_menu", show_menu)
    graph.add_node("fetch_joke", fetch_joke)
    graph.add_node("update_category", update_category)
    graph.add_node("update_language", update_language)
    graph.add_node("reset_jokes", reset_jokes)
    graph.add_node("exit_bot", exit_bot)
    
    graph.set_entry_point("show_menu")
    
    graph.add_conditional_edges("show_menu",route_choice, {
        "fetch_joke": "fetch_joke",
        "update_category": "update_category",
        "update_language": "update_language",
        "reset_jokes": "reset_jokes",
        "exit_bot": "exit_bot"
    })
    graph.add_edge("fetch_joke", "show_menu")
    graph.add_edge("update_category", "show_menu")
    graph.add_edge("update_language", "show_menu")
    graph.add_edge("reset_jokes", "show_menu")
    graph.add_edge("exit_bot", END)
    
    return graph.compile()

def main():
    graph = build_joke_graph()
    
    # print(graph.get_graph().draw_mermaid())
    # graph.get_graph().print_ascii()
    
    final_state = graph.invoke(Joke_State(), config={"recursion_limit": 100})
    
if __name__ == "__main__":
    print("")
    print("🎉===========================================================🎉")
    print("               WELCOME TO THE LANGGRAPH JOKE BOT!")
    print("   This example demonstrates agentic state flow without LLMs")
    print("  ===========================================================")
    print("")
    print("🚀===========================================================🚀")
    print("                  STARTING JOKE BOT SESSION...")
    main()