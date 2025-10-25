from langgraph.graph import StateGraph
import asyncio
from pydantic import BaseModel
from typing import Annotated, List
from operator import add
from gazetteer_load_extract import load_gazetteer_json, extract_from_gazetteer
import spacy
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv

load_dotenv()

class Tag(BaseModel):
    user_input: str
    tags: List[str]

def start_input(State: Tag):
    txt = input("please provide the publication")
    return {"txt": txt}

def gazetteer_extraction(text: str):
    gazetteer = load_gazetteer_json("data/gazetteer.json")
    tag = extract_from_gazetteer(text, gazetteer)
    print(tag)
    return tag

# ml extraction, tag extratiojn using spacy model 
def spacy_extraction(txt: str):
    nnl = spacy.load("en_core_web_md")
    docs = nnl(txt)
    print(docs)
    for doc in docs:
        print("this is spacy doc: ", doc)
 
# tag extration using llm   
def llm_extraction(text: str):
    prompt =f"""
                You are an expert AI text tagger. 
                Given the text below, identify and extract relevant technical tags such as AI terms, software frameworks, programming languages, and architectures. 
                Return the result as a JSON list of tags (each tag should include `canonical_name`, `category`, and `confidence` score). 
                Focus only on meaningful, domain-specific tags — avoid common words or duplicates. 
                # and return the original text you being working on after the tag extraction
                Text: "{text}"
            """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature="0.2")
    llm_tags = llm.invoke(prompt)
    # print("this is llm extraction node", llm_tags.content)

# get all three tag  extratction and merges
def llm_aggregation():
    print("this is llm aggregate node")

# print the tag
def print_tag():
    print("these are the tag", )
    
async def paralle_extraction(State: Tag):
    result = await asyncio.gather(
        gazetteer_extraction(State.user_input),
        spacy_extraction(State.user_input),
        llm_extraction(State.user_input)
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