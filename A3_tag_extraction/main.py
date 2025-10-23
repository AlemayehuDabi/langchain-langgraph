from langgraph.graph import StateGraph
import asyncio
from pydantic import BaseModel
from typing import Annotated
from operator import add


class Tag(BaseModel):
    txt: str

def start_input(State: Tag):
    txt = input("please provide the publication")
    return {"txt": txt}

def gazetteer_extraction(State: Tag):
    print("this is the gazetteer node")

def llm_extraction():
    print("this is llm extraction node")
    
def spacy_extraction():
    print("this ml extraction node")

def llm_aggregation():
    print("this is llm aggregate node")

def print_tag():
    print("these are the tag", )
    
async def paralle_extraction():
    result = await asyncio.gather(
        gazetteer_extraction(),
        spacy_extraction(),
        llm_extraction()
    )
    
    gazetteer_text, spacy_text, llm_text = result
    return {
        "gazetteer_text": gazetteer_text,
        "spacy_text": spacy_text,
        "llm_text": llm_text
    } 

def garphCondition():
    graph = StateGraph()  
    
    graph.set_entry_point("start_input", start_input)
    graph.add_node("paralle_extraction", paralle_extraction)
    graph.add_node("llm_agregation", llm_aggregation)
    graph.add_node("print_tag", print_tag)
    
    graph.add_edge("start_input", "paralle_extraction")
    graph.add_edge("paralle_extraction", "llm_agregation")
    graph.add_edge("llm_agregation", "print_tag")