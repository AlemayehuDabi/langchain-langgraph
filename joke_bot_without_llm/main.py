from typing import Annotated, Literal
from pydantic import BaseModel
from pyjokes import get_joke
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from operator import add


class Joke(BaseModel):
    txt: str
    joke_category: str
    
class Joke_State(BaseModel):
    jokes: Annotated[list[Joke], add] = []
    joke_choice: Literal["n", "c", "q"] = "n"
    category: str = "neutral"
    language: str = "en"
    quit: bool = False
    
def show_menu(state: Joke_State) -> dict:
    print("============================================================")
    print(f"🎭 Menu | Category: {state.category} | Jokes: {len(state.jokes)}")
    print("--------------------------------------------------")
    print("Pick an option:")
    user_input = input("[n] 🎭 Next Joke  [c] 📂 Change Category  [q] 🚪 Quit").strip().lower()
    print("user_name: ", user_input)
    return { "joke_choice": user_input }

def fetch_joke(state: Joke_State) -> dict:
    joke = get_joke(state.language, state.category)
    new_joke = Joke(txt=joke, joke_category=state.category)
    print(f"\n😂 {new_joke.txt}\n")
    return { "jokes": [new_joke] }

def update_category(state: Joke_State) -> dict:
    categories = ["neutral", "chuck", "all"]
    selected_input = int(input("[0] - netural, [1] - chuck, [2] - all").strip())
    return {"category": categories[selected_input]} 

def exit_bot(state: Joke_State) -> dict:
    return { "quit": True}

def route_choice(state: Joke_State) -> str:
    if state.joke_choice == "n":
        return "fetch_joke"
    elif state.joke_choice == "c":
        return "update_category"
    elif state.joke_choice == "q":
        return "exit_bot"
    return "exit_bot"

def build_joke_graph() -> CompiledStateGraph:
    graph = StateGraph(Joke_State)
    
    graph.add_node("show_menu", show_menu)
    graph.add_node("fetch_joke", fetch_joke)
    graph.add_node("update_category", update_category)
    graph.add_node("exit_bot", exit_bot)
    
    graph.set_entry_point("show_menu")
    
    graph.add_conditional_edges("show_menu",route_choice, {
        "fetch_joke": "fetch_joke",
        "update_category": "update_category",
        "exit_bot": "exit_bot"
    })
    graph.add_edge("fetch_joke", "show_menu")
    graph.add_edge("update_category", "show_menu")
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