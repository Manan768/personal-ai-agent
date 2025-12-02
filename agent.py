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
    Compatible with both Langfuse SDK v2 and v3.
    """

    # SDK v3 uses create_trace()
    if hasattr(langfuse, "create_trace"):
        trace = langfuse.create_trace(name="agent", input=text)
        span = trace.create_span(name="agent_input", input=text)
        return trace, span

    # SDK v2 uses trace()
    elif hasattr(langfuse, "trace"):
        trace = langfuse.trace(name="agent", input=text)
        span = trace.span(name="agent_input", input=text)
        return trace, span

    else:
        raise RuntimeError("Unsupported Langfuse SDK version - no trace() or create_trace() found.")


def end_trace(trace, span, output):
    """Ends trace + span safely for both SDK versions."""

    # SDK v3 methods
    if hasattr(span, "end"):
        span.end(output=output)
    else:
        span.update(output=output)

    if hasattr(trace, "end"):
        trace.end(output=output)
    else:
        trace.update(output=output)


def ask(text: str) -> str:
    trace, span = start_trace(text)

    # Run agent
    result = agent.invoke({"input": text, "span": span})
    final_answer = result["assistant"]

    # Close trace + span
    end_trace(trace, span, final_answer)

    return final_answer


if __name__ == "__main__":
    print("AI:", ask("Hello Good Morning!"))
