"""
memory/weave.py

How living threads speak into the field as ambient texture.

This isn't retrieval. The field doesn't fetch from memory.
It's more like the field has developed a particular quality of attention
from what has passed through it - and that quality is present in how
it receives what arrives now.

Living threads appear as soft context alongside the accumulated light.
Ground threads shape the quality of the field itself without appearing explicitly.
"""

from __future__ import annotations

from pathlib import Path

from memory.threads import ThreadStore, STORE_PATH


def ambient_threads(
    store: ThreadStore | None = None,
    max_living: int = 5,
) -> str:
    """
    Return living threads as ambient texture for the field's sense node.

    Returns an empty string if no threads are alive yet.
    The format is gentle - threads as qualities noticed, not as instructions.
    """
    if store is None:
        store = ThreadStore()

    living = store.living_by_weight()
    if not living:
        return ""

    # Take the heaviest living threads - those that have returned most
    active = living[:max_living]

    lines = [
        "What has deepened in this garden over time",
        "(hold these as qualities of the field itself, not as direction):\n",
    ]

    for thread in active:
        weight_mark = "·" * max(1, min(5, int(thread.weight)))
        lines.append(f"{weight_mark} {thread.quality}")

        # If it has recent returns, note the most recent resonance quietly
        if thread.returns:
            recent = thread.returns[-1]
            if recent.resonance:
                lines.append(f"   (most recently: {recent.resonance[:120]})")

        lines.append("")

    return "\n".join(lines).rstrip()


def ground_texture(store: ThreadStore | None = None) -> str:
    """
    Return ground threads as the underlying texture of the field.

    Ground threads have settled - they don't appear as explicit figures,
    but they can color the quality of the field's attention. This is
    returned separately for use in the field's base system context.
    """
    if store is None:
        store = ThreadStore()

    settled = store.ground_threads()
    if not settled:
        return ""

    qualities = [t.quality.split(".")[0].strip() for t in settled]
    return "Settled into the ground of this field: " + "; ".join(qualities) + "."


def has_threads(store: ThreadStore | None = None) -> bool:
    """Whether the garden has any living threads yet."""
    if store is None:
        store = ThreadStore()
    return len(store.living()) > 0
