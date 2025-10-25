from langgraph.graph import StateGraph
import asyncio
from pydantic import BaseModel
from typing import Annotated
from operator import add
from gazetteer_load_extract import load_gazetteer_json, extract_from_gazetteer
import spacy

class Tag(BaseModel):
    txt: str

def start_input(State: Tag):
    txt = input("please provide the publication")
    return {"txt": txt}

def gazetteer_extraction(text: str):
    gazetteer = load_gazetteer_json("data/gazetteer.json")
    tag = extract_from_gazetteer(text, gazetteer)
    print(tag)
    return tag


    
# ml extraction, tag extratiojn using spacy model 
def spacy_extraction():
    nnl = spacy.load("en_core_web_md")
    txt="Machine learning and AI are part of modern software."
    docs = nnl(txt)
    for doc in docs:
        print("this is spacy doc: ", doc)
 
# tag extration using llm   
def llm_extraction():
    print("this is llm extraction node")

# get all three tag  extratction and merges
def llm_aggregation():
    print("this is llm aggregate node")

# print the tag
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
    graph.add_node("llm_agregation", llm_aggregation)
    graph.add_node("print_tag", print_tag)
    
    graph.add_edge("start_input", "llm_agregation")
    graph.add_edge("llm_agregation", "print_tag")