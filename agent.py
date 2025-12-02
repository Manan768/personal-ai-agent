import os
from langfuse import Langfuse
from graph import build_graph

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

agent = build_graph()

def start_trace(text: str):
    """
    Updated for latest Langfuse SDK — always uses trace() and span().
    """
    try:
        trace = langfuse.trace(name="agent", input=text)
        span = trace.span(name="agent_input", input=text)
        return trace, span
    except Exception as e:
        # If SDK does not support tracing, fallback gracefully
        print("⚠ Langfuse tracing disabled due to:", e)
        return None, None


def end_trace(trace, span, output):
    """Safely close span + trace if they exist."""
    if span:
        try:
            span.end(output=output)
        except:
            pass

    if trace:
        try:
            trace.end(output=output)
        except:
            pass


def ask(text: str) -> str:
    trace, span = start_trace(text)

    result = agent.invoke({"input": text, "span": span})
    final_answer = result["assistant"]

    end_trace(trace, span, final_answer)
    return final_answer


if __name__ == "__main__":
    print("AI:", ask("Hello Good Morning!"))
