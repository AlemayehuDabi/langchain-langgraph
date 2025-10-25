from langgraph.graph import StateGraph
import asyncio
from pydantic import BaseModel
from typing import List
from gazetteer_load_extract import load_gazetteer_json, extract_from_gazetteer
import spacy
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
import json
import re

load_dotenv()

# keep your same LLM instance (invoke is synchronous so we'll call it via to_thread)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature="0.2")

# -------------------------
# State model (kept shape, added safe defaults)
# -------------------------
class Tag(BaseModel):
    user_input: str = ""
    tags: List[str] = []
    gazetteer_text: List[str] = []
    spacy_text: List[str] = []
    llm_text: List[str] = []

# -------------------------
# Start input (unchanged)
# -------------------------
def start_input(State: Tag):
    txt = input("please provide the publication")
    return {"user_input": txt}

# -------------------------
# Helper: parse JSON from LLM text (strip code fences, etc.)
# -------------------------
def parse_llm_json(text: str):
    if not text:
        return None
    # remove triple backticks and optional "json" tag
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    # sometimes the model returns extra text before/after JSON; try to find a JSON array/object
    m = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1)
    try:
        return json.loads(cleaned)
    except Exception:
        # last fallback: try to eval-like parsing replacements (dangerous generally; here safe fallback)
        try:
            # Replace single quotes with double quotes (naive)
            alt = cleaned.replace("'", '"')
            return json.loads(alt)
        except Exception:
            return None

# -------------------------
# Gazetteer extraction (async wrapper)
# -------------------------
async def gazetteer_extraction(text: str):
    # run load & extract in thread (both are sync)
    def _sync_gazetteer_extract(t):
        gaz = load_gazetteer_json("data/gazetteer.json")
        return extract_from_gazetteer(t, gaz)

    raw = await asyncio.to_thread(_sync_gazetteer_extract, text)

    # normalize to list[str] (extract matched text or canonical)
    normalized = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                # prefer matched key, else canonical, else stringified dict
                val = item.get("matched") or item.get("canonical") or json.dumps(item, ensure_ascii=False)
                normalized.append(str(val))
            else:
                normalized.append(str(item))
    else:
        # unexpected shape -> stringify
        normalized = [str(raw)]
    return normalized

# -------------------------
# spaCy extraction (async wrapper)
# -------------------------
# load the model once globally to avoid repeated heavy loads
# using "en_core_web_md" as your code used it; fallback to small if not available.
try:
    nlp = spacy.load("en_core_web_md")
except Exception:
    # best-effort fallback
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # if even that fails, set nlp to None and handle later
        nlp = None

async def spacy_extraction(txt: str):
    if nlp is None:
        return []

    # run the actual nlp doc creation in a thread
    doc = await asyncio.to_thread(nlp, txt)

    # prefer named entities (doc.ents) — fallback to noun_chunks or tokens
    results = []
    if doc.ents:
        for ent in doc.ents:
            results.append(ent.text)
    else:
        # use noun_chunks if available
        nchunks = [chunk.text for chunk in doc.noun_chunks]
        if nchunks:
            results.extend(nchunks)
        else:
            # last resort: tokens as strings
            results.extend([tok.text for tok in doc])

    # normalize and dedupe while preserving order
    seen = set()
    normalized = []
    for t in results:
        s = str(t).strip()
        if s and s not in seen:
            seen.add(s)
            normalized.append(s)
    return normalized

# -------------------------
# LLM extraction (async wrapper)
# -------------------------
async def llm_extraction(text: str):
    prompt = f"""
    You are an expert AI text tagger.
    Given the text below, identify and extract relevant technical tags such as AI terms, software frameworks, programming languages, and architectures.
    Return the result as a JSON list of tags (each tag should include `canonical_name`, `category`, and `confidence` score).
    Focus only on meaningful, domain-specific tags — avoid common words or duplicates.
    Text: "{text}"
    """

    # llm.invoke is synchronous — call in a thread
    resp = await asyncio.to_thread(llm.invoke, prompt)
    # resp.content expected to be string
    content = getattr(resp, "content", resp) if resp is not None else ""
    parsed = parse_llm_json(str(content))

    # normalized list[str] of canonical_name (or raw strings if structure unknown)
    normalized = []
    if isinstance(parsed, list):
        for entry in parsed:
            if isinstance(entry, dict):
                cn = entry.get("canonical_name") or entry.get("canonical") or entry.get("name") or entry.get("tag")
                if cn:
                    normalized.append(str(cn))
                else:
                    # fallback: serialize entry to string
                    normalized.append(json.dumps(entry, ensure_ascii=False))
            else:
                normalized.append(str(entry))
    elif isinstance(parsed, dict):
        # maybe the model returned an object with "tags" key
        tags = parsed.get("tags") or parsed.get("merged_tags") or parsed.get("results")
        if isinstance(tags, list):
            for entry in tags:
                if isinstance(entry, dict):
                    cn = entry.get("canonical_name") or entry.get("canonical") or entry.get("name")
                    normalized.append(str(cn) if cn else json.dumps(entry, ensure_ascii=False))
                else:
                    normalized.append(str(entry))
        else:
            # unknown shape -> fallback to raw content
            normalized = [str(content)]
    else:
        # could not parse JSON -> maybe raw text containing tag list; try splitting lines
        lines = [ln.strip() for ln in str(content).splitlines() if ln.strip()]
        # keep only short lines
        normalized = [ln for ln in lines if len(ln) < 200][:50] or [str(content)]

    # dedupe preserve order
    seen = set()
    out = []
    for t in normalized:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# -------------------------
# Aggregation/refinement (async)
# -------------------------
async def llm_aggregation(State: Tag):
    # Build a safe, small prompt that passes the lists (strings)
    prompt = f"""
    You are an expert AI tag aggregator. Merge and refine tags from three sources: gazetteer, spaCy, and LLM.
    Remove duplicates, resolve inconsistencies, and discard hallucinated tags by validating with the original text.
    Keep only meaningful technical tags (AI terms, software frameworks, programming languages, architectures).
    Return JSON with a single field `merged_tags` which is a list of objects each containing `canonical_name`, `category`, and `confidence` (0.0-1.0).
    Gazetteer tags: {State.gazetteer_text}
    spaCy tags: {State.spacy_text}
    LLM tags: {State.llm_text}
    Original text: "{State.user_input}"
    """

    resp = await asyncio.to_thread(llm.invoke, prompt)
    content = getattr(resp, "content", resp) if resp is not None else ""
    parsed = parse_llm_json(str(content))

    merged = []
    if isinstance(parsed, dict) and isinstance(parsed.get("merged_tags"), list):
        for item in parsed["merged_tags"]:
            if isinstance(item, dict):
                cn = item.get("canonical_name") or item.get("canonical") or item.get("name")
                if cn:
                    merged.append(str(cn))
    elif isinstance(parsed, list):
        # if model returned list of tag-objects
        for item in parsed:
            if isinstance(item, dict):
                cn = item.get("canonical_name") or item.get("canonical") or item.get("name")
                if cn:
                    merged.append(str(cn))
    else:
        # fallback: parse lines from raw content
        lines = [ln.strip() for ln in str(content).splitlines() if ln.strip()]
        for ln in lines:
            # ignore JSON markers and simple punctuation
            if ln.startswith("{") or ln.startswith("[") or ln.endswith("}") or ln.endswith("]"):
                continue
            merged.append(ln)

    # final dedupe & shorten to unique canonical strings
    seen = set()
    final_tags = []
    for t in merged:
        if t not in seen:
            seen.add(t)
            final_tags.append(t)

    # return matching shape for your Tag.tags (List[str])
    return {"tags": final_tags}

# -------------------------
# Print tags (async)
# -------------------------
async def print_tag(State: Tag):
    print("these are the tag", State.tags)

# -------------------------
# Parallel extraction node (async)
# -------------------------
async def paralle_extraction(State: Tag):
    # gather three async extraction functions
    gazetteer_text, spacy_text, llm_text = await asyncio.gather(
        gazetteer_extraction(State.user_input),
        spacy_extraction(State.user_input),
        llm_extraction(State.user_input)
    )

    # return fields that match your Tag model (list[str])
    return {
        "gazetteer_text": gazetteer_text,
        "spacy_text": spacy_text,
        "llm_text": llm_text
    }

# -------------------------
# Graph setup (fixed node name typo)
# -------------------------
def graphCondition():
    graph = StateGraph(Tag)

    graph.add_node("start_input", start_input)
    graph.add_node("paralle_extraction", paralle_extraction)
    graph.add_node("llm_aggregation", llm_aggregation)   # fixed name
    graph.add_node("print_tag", print_tag)

    graph.set_entry_point("start_input")
    graph.add_edge("start_input", "paralle_extraction")
    graph.add_edge("paralle_extraction", "llm_aggregation")
    graph.add_edge("llm_aggregation", "print_tag")

    return graph.compile()

# -------------------------
# main (uses async API)
# -------------------------
async def main():
    app = graphCondition()

    # Use empty dict so start_input runs and populates user_input
    final_state = await app.ainvoke({}, config={"recursion_limit": 100})

    # final_state is a Tag model instance; print to confirm
    try:
        print("\nFINAL STATE:")
        print("user_input:", final_state.user_input)
        print("gazetteer_text:", final_state.gazetteer_text)
        print("spacy_text:", final_state.spacy_text)
        print("llm_text:", final_state.llm_text)
        print("merged tags:", final_state.tags)
    except Exception as e:
        print("Could not pretty-print final_state:", e)

if __name__ == "__main__":
    asyncio.run(main())


# Machine learning and AI are part of modern software.