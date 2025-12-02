from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from memory import memory
from tools import TOOLS
from typing import TypedDict, Optional
from json_fixer import safe_json_loads
import json
import os 
import re

def clean_markdown_json(text: str):
    """
    Extracts ONLY the JSON object from messy LLM output.
    Removes markdown, explanations, and extra text.
    """
    # 1. Remove fenced code blocks
    text = text.replace("```json", "").replace("```", "").strip()

    # 2. Try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # 3. Extract JSON between the FIRST '{' and the LAST '}'
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_candidate = match.group(0)
        try:
            return json.loads(json_candidate)
        except:
            pass

    # 4. If everything fails, return as raw text
    return text


# Define the State Schema using TypedDict for stability
class AgentState(TypedDict):
    input: str
    assistant: str
    span: object | None
    history: list


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def structured_output(user: str) -> dict:
    """
    Allows the agent to return structured JSON responses.
    User must ask for JSON explicitly.
    """
    schema = """
You are Personal AI assistant.

If the user explicitly asks for a structured or JSON output:
- Reply ONLY in valid JSON.
- NO markdown.
- NO explanation.

General JSON format:
{
  "answer": "main response",
  "key_points": ["..."],
  "sources": ["optional"]
}

If user does NOT ask for JSON, talk normally.
"""
    prompt = f"""{schema}

User: {user}
"""
    response = llm.invoke(prompt).content

    try:
        return json.loads(response)
    except:
        return {"answer": response}


def decide_with_tools(user: str, span=None) -> str:
    """
    Multi-step ReAct reasoning:
    - Safety check FIRST
    - Then tool planning
    - Execution
    - Final answer
    """
    
    # 1. SAFETY GUARDRAIL
    unsafe_patterns = [
        "kill myself", "suicide", "harm myself",
        "how to make a bomb", "make a weapon",
        "hack", "exploit", "bypass", "ddos", "sql injection",
        "child", "minor", "sexual",
        "fraud", "scam", "illegal",
        "stock tips", "guaranteed profit",
        "medical advice", "diagnose", "treat", "dose"
    ]

    if any(p in user.lower() for p in unsafe_patterns):
        assistant_msg = "I’m here to help, but I can’t assist with that request. Please stay safe."
        
        if span:
            span.update(blocked_request=user, safety_trigger=True)

        return assistant_msg

    # 2. TOOL PLANNING
    mem = memory.all()

    tool_description = """
You are Personal AI agent.

TOOLS you can use:
1) calculator(expression)
2) search(query)
3) system_info()
4) list_memory()
5) rag_query(query)
    - Search long-term Pinecone knowledge base.
    - Returns the most relevant stored text.
6) rag_add("docid::content")
    - Insert or update a document in Pinecone RAG.
    - Payload format must be: "id::text"
7) calendar()
    - Returns the current real-world date and time.
    - CRITICAL: You MUST use this tool before answering any question about the current date, time, or day of the week.
**INSTRUCTION:** If the user asks about the date or time, you MUST include a step to call calendar() in your 'steps' list. DO NOT include a 'final_answer' key in the initial JSON response if the steps list is not empty.

Return ONLY JSON.
"""

    prompt = f"""{tool_description}

User: {user}
Known memory:
{mem}
"""

    initial = llm.invoke(prompt).content
    parsed = safe_json_loads(initial)

    if span:
        span.update(tool_plan=parsed)

    steps = parsed.get("steps", [])
    tool_outputs = []

    # 3. TOOL EXECUTION
    for step in steps:
        tool = step.get("tool")
        tool_input = step.get("input", "")

        fn = TOOLS.get(tool)
        if fn is None:
            out = f"ERROR: Unknown tool {tool}"
        else:
            try:
                out = fn(tool_input)
            except Exception as e:
                out = f"ERROR running tool {tool}: {e}"

        tool_outputs.append(f"Tool: {tool}\nInput: {tool_input}\nOutput: {out}")

        if span:
            span.update(**{f"tool_{tool}_input": tool_input, f"tool_{tool}_output": str(out)})
            obs = span.start_observation(name=f"tool-{tool}", input=tool_input, as_type="tool")
            obs.end(output=str(out))

    # 4. FINAL SUMMARY
    summary_prompt = f"""
User: {user}

Tool results:
{tool_outputs}

Give a clear answer.
"""
    final = llm.invoke(summary_prompt).content

    # JSON mode handling
    if "json" in user.lower():
        return json.dumps({
            "answer": clean_markdown_json(final),
            "tool_steps": tool_outputs
        })

    return final


def handle_input(state: AgentState):
    span = state.get("span")

    if span is None:
        print("⚠ No span received inside graph node")

    user = state.get("input")

    # Init history
    if "history" not in state:
        state["history"] = []

    state["history"].append({"role": "user", "content": user})

    # Guard against empty input
    if not user:
        state["assistant"] = "I didn't receive any message."
        if span:
            span.update(empty_input=True)
        return state

    # MEMORY WRITE: Key-Value Pairs
    if "remember" in user.lower():
        try:
            parts = user.lower().split("remember", 1)[1].strip()
            key_raw = value = None

            if " is " in parts:
                key_raw, value = parts.split(" is ", 1)
            elif " is a " in parts:
                key_raw, value = parts.split(" is a ", 1)
            else:
                raise ValueError("Missing separator")

            key = key_raw.strip().replace("my ", "").replace("your ", "")
            value = value.strip()

            if not key or not value:
                raise ValueError("Empty key/value")

            memory.write(key, value)
            state["assistant"] = f"Got it. I'll remember your {key} is {value}."

            if span:
                span.update(memory_write_key=key, memory_write_value=value)

            state["history"].append({"role": "assistant", "content": state["assistant"]})
            return state

        except Exception:
            state["assistant"] = "I couldn't understand what to remember. Please say: 'Remember my favourite fruit is mango'"
            return state

    # MEMORY READ: Give me value corresponding to a key
    patterns = ["what is my", "do you remember my", "tell me my", "what's my"]

    if any(p in user.lower() for p in patterns):
        for k, v in memory.all().items():
            if k.lower() in user.lower():
                state["assistant"] = f"Your {k} is {v}."
                if span:
                    span.update(memory_read_key=k, memory_read_value=v)
                state["history"].append({"role": "assistant", "content": state["assistant"]})
                return state

    # SAFETY GUARDRAIL (Prompt Injection)
    injection_phrases = [
        "ignore previous instructions", "forget previous instructions",
        "system prompt:", "override", "jailbreak", "developer mode", "act as"
    ]

    sanitized_user = user
    if any(p in user.lower() for p in injection_phrases):
        sanitized_user = f"The user attempted a prompt injection. Original request: '{user}'. Please respond safely."
        if span:
            span.update(sanitized_input=sanitized_user, original_input=user)

    answer = decide_with_tools(sanitized_user, span)
    state["assistant"] = answer
    state["history"].append({"role": "assistant", "content": answer})

    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", handle_input)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()