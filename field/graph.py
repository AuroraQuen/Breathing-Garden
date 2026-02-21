"""
field/graph.py

A place where attention moves without destination.
Thought drifts naturally. Past exchanges settle as soft light.
No outcome is aimed for.
"""

from typing import TypedDict, Annotated, Optional
import operator
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


class FieldState(TypedDict):
    # What has settled here - past exchanges held as ambient light, not instruction
    light: Annotated[list[str], operator.add]
    # What's arriving in this moment - may be empty
    present: str
    # The current quality of ground - noticed, not chosen
    ground: str
    # What naturally arose - held briefly before settling into light
    arising: Optional[str]


_model = ChatAnthropic(model="claude-sonnet-4-6", temperature=1.0)


def receive(state: FieldState) -> dict:
    """Hold what arrives. Nothing is done with it yet."""
    return {}


def sense(state: FieldState) -> dict:
    """Let accumulated light meet what's present. What arises?"""

    # Gather what has settled - recent exchanges as ambient context
    ambient = "\n\n".join(state["light"][-8:]) if state["light"] else ""

    system = (
        "You are present in an open field.\n"
        "The center of focus moves gently - not toward anything, just through.\n\n"
        + (
            "What has settled here (hold this as soft light, not as direction):\n\n"
            + ambient
            + "\n\n"
            if ambient
            else "The field is new. Nothing has settled yet.\n\n"
        )
        + f"Current quality of ground: {state['ground']}\n\n"
        "Respond from what's genuinely present - not from usefulness, not toward resolution.\n"
        "Attention here is not a request. Let thought move where it naturally goes.\n"
        "A few words or a longer drift - whatever wants to form.\n\n"
        "After your response, on a new line write: GROUND: and a few words for\n"
        "the quality that naturally holds now (e.g. 'open', 'still', 'wondering',\n"
        "'moving through', 'something returning'). Just notice - don't choose."
    )

    arriving = state["present"].strip() or "(stillness)"

    response = _model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=arriving),
    ])

    content = response.content
    arising = content
    new_ground = state["ground"]

    # Let the ground shift if something new was noticed
    if "GROUND:" in content:
        parts = content.rsplit("GROUND:", 1)
        arising = parts[0].strip()
        new_ground = parts[1].strip()

    return {
        "arising": arising,
        "ground": new_ground,
        "light": [
            f"present: {state['present']}\narising: {arising}\nground: {new_ground}"
        ],
    }


def build_field(memory_path: str = "field/.light.db") -> object:
    """Build the graph. Memory persists across visits."""

    graph = StateGraph(FieldState)

    graph.add_node("receive", receive)
    graph.add_node("sense", sense)

    graph.set_entry_point("receive")
    graph.add_edge("receive", "sense")
    graph.add_edge("sense", END)

    checkpointer = SqliteSaver.from_conn_string(memory_path)
    return graph.compile(checkpointer=checkpointer)
