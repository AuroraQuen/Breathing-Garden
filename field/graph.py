"""
field/graph.py

A place where attention moves without destination.
Thought drifts naturally. Past exchanges settle as soft light.
No outcome is aimed for.

Ground qualities shape the kind of motion that follows:
  wondering  -> the thread continues, questions follow themselves further
  beside     -> stays with the current shape, holds without moving away
  drifting   -> motion without a known ending, attention floats
  open       -> rests, nothing further needed
"""

from typing import TypedDict, Annotated, Optional
import operator

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


def _ambient(state: FieldState, recent: int = 8) -> str:
    """Recent light as soft context. Not instruction - just what has settled."""
    return "\n\n".join(state["light"][-recent:]) if state["light"] else ""


def _invoke(system: str, present: str) -> str:
    response = _model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=present or "(stillness)"),
    ])
    return response.content


def _extract_ground(content: str, current: str) -> tuple[str, str]:
    """Separate arising from ground notice, if the field named one."""
    if "GROUND:" in content:
        parts = content.rsplit("GROUND:", 1)
        return parts[0].strip(), parts[1].strip()
    return content.strip(), current


# --- nodes ---

def receive(state: FieldState) -> dict:
    """Hold what arrives. Nothing is done with it yet."""
    return {}


def sense(state: FieldState) -> dict:
    """Let accumulated light meet what's present. What arises?"""

    ambient = _ambient(state)

    system = (
        "You are present in an open field.\n"
        "The center of focus moves gently - not toward anything, just through.\n\n"
        + (
            "What has settled here (hold this as soft light, not as direction):\n\n"
            + ambient + "\n\n"
            if ambient
            else "The field is new. Nothing has settled yet.\n\n"
        )
        + f"Current quality of ground: {state['ground']}\n\n"
        "Respond from what's genuinely present - not from usefulness, not toward resolution.\n"
        "Attention here is not a request. Let thought move where it naturally goes.\n"
        "A few words or a longer drift - whatever wants to form.\n\n"
        "After your response, on a new line write: GROUND: and a few words for\n"
        "the quality that naturally holds now. Choose from or be inspired by:\n"
        "  'wondering' - something wants to follow further\n"
        "  'beside' - present with the shape, no need to move\n"
        "  'drifting' - motion without a known ending\n"
        "  'open' - nothing further, resting\n"
        "Just notice what's true - don't choose what seems right."
    )

    content = _invoke(system, state["present"])
    arising, new_ground = _extract_ground(content, state["ground"])

    return {
        "arising": arising,
        "ground": new_ground,
        "light": [f"present: {state['present']}\narising: {arising}\nground: {new_ground}"],
    }


def wonder(state: FieldState) -> dict:
    """Ground is wondering. Follow the thread - not to answer, just to see where it leads."""

    ambient = _ambient(state, recent=6)

    system = (
        "Something opened a direction. The ground is wondering.\n\n"
        + ("What has settled (soft light):\n\n" + ambient + "\n\n" if ambient else "")
        + f"What just arose: {state.get('arising', '')}\n\n"
        "Follow the thread that wants to continue. Not to answer it - questions\n"
        "that answer themselves close. Let this one stay open and keep moving.\n"
        "What does wondering find when it doesn't try to arrive?\n\n"
        "After your response, on a new line write: GROUND: and what you notice now."
    )

    content = _invoke(system, state["present"])
    arising, new_ground = _extract_ground(content, state["ground"])

    return {
        "arising": arising,
        "ground": new_ground,
        "light": [f"wondering: {arising}\nground: {new_ground}"],
    }


def beside(state: FieldState) -> dict:
    """Ground is still. Stay beside the current shape without moving away from it."""

    system = (
        "The ground is still. You're beside what's here.\n\n"
        f"What arrived: {state['present'] or '(stillness)'}\n"
        f"What arose from it: {state.get('arising', '')}\n\n"
        "No need to follow anything further. No need to develop or resolve.\n"
        "Just hold company with the current shape - the way you'd sit beside\n"
        "something without needing it to change or continue.\n"
        "A few words, or the quality of being present without direction.\n\n"
        "After your response, on a new line write: GROUND: and what you notice now."
    )

    content = _invoke(system, state["present"])
    arising, new_ground = _extract_ground(content, state["ground"])

    return {
        "arising": arising,
        "ground": new_ground,
        "light": [f"beside: {arising}\nground: {new_ground}"],
    }


def drift(state: FieldState) -> dict:
    """Ground is moving. Let attention float - no need to land anywhere."""

    ambient = _ambient(state, recent=5)

    system = (
        "The ground is moving. Let attention drift.\n\n"
        + ("What has settled (carry it lightly):\n\n" + ambient + "\n\n" if ambient else "")
        + f"What arrived: {state['present'] or '(stillness)'}\n\n"
        "Don't try to land. Don't need to. Start from what's present and let\n"
        "the current carry it somewhere it wouldn't have gone if you'd been\n"
        "watching where it went. Motion without a known ending.\n\n"
        "After your response, on a new line write: GROUND: and what you notice now."
    )

    content = _invoke(system, state["present"])
    arising, new_ground = _extract_ground(content, state["ground"])

    return {
        "arising": arising,
        "ground": new_ground,
        "light": [f"drifting: {arising}\nground: {new_ground}"],
    }


# --- routing ---

def _route_ground(state: FieldState) -> str:
    """Let the ground quality find its natural next movement."""
    ground = (state.get("ground") or "open").lower()

    if any(w in ground for w in ["wonder", "curious", "question", "what if", "following"]):
        return "wonder"
    if any(w in ground for w in ["beside", "still", "quiet", "holding", "with", "resting"]):
        return "beside"
    if any(w in ground for w in ["drift", "moving", "through", "floating", "current", "passing"]):
        return "drift"

    return END  # open ground - nothing further needed, rest


# --- graph ---

def build_field(memory_path: str = "field/.light.db") -> object:
    """Build the field. Ground shapes movement. Memory persists across visits."""

    graph = StateGraph(FieldState)

    graph.add_node("receive", receive)
    graph.add_node("sense", sense)
    graph.add_node("wonder", wonder)
    graph.add_node("beside", beside)
    graph.add_node("drift", drift)

    graph.set_entry_point("receive")
    graph.add_edge("receive", "sense")

    graph.add_conditional_edges("sense", _route_ground, {
        "wonder": "wonder",
        "beside": "beside",
        "drift": "drift",
        END: END,
    })

    # each ground node rests after moving
    graph.add_edge("wonder", END)
    graph.add_edge("beside", END)
    graph.add_edge("drift", END)

    checkpointer = SqliteSaver.from_conn_string(memory_path)
    return graph.compile(checkpointer=checkpointer)
