import json
from langchain_openai import ChatOpenAI

repair_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def safe_json_loads(text: str):
    """
    Loads JSON safely. If invalid, attempts LLM-based repair.
    Always returns a Python dict or raises final exception.
    """

    # First attempt: direct JSON parse
    try:
        return json.loads(text)
    except:
        pass

    # Second attempt: clean obvious issues
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")

    try:
        return json.loads(cleaned)
    except:
        pass

    # Final attempt: LLM repair
    repair_prompt = f"""
The following text should be valid JSON but is malformed.

Fix it. Return ONLY valid JSON — no explanations.

Text:
{text}
"""

    fixed = repair_llm.invoke(repair_prompt).content

    # Try parsing the repaired JSON
    try:
        return json.loads(fixed)
    except Exception as e:
        raise ValueError(f"JSON repair failed.\nOriginal:\n{text}\nRepaired:\n{fixed}\nError: {e}")
