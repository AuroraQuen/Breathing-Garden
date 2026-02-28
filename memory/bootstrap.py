"""
memory/bootstrap.py

Reads written moments and surfaces threads from them - used when the garden
is new, or when new moments have been added and you want threads to recognize
what's accumulated in writing rather than only in the field's own motion.

The field notices threads on its own during sessions (via the remember node).
This is for seeding from what has been written - moments, spaces, reflections.

Usage:
    python -m memory.bootstrap                    # read all moments
    python -m memory.bootstrap --field-thread t   # also read a named field thread
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from memory.threads import ThreadStore, Thread, Return, new_thread, STORE_PATH, SETTLE_WEIGHT


MOMENTS_DIR = Path(__file__).parent.parent / "moments"
_model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)


def _invoke(system: str, content: str) -> str:
    response = _model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=content),
    ])
    return response.content


def _read_moments() -> list[tuple[str, str]]:
    """Read all moment files. Returns (source_name, content) pairs."""
    moments = []
    for path in sorted(MOMENTS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        moments.append((f"moments/{path.name}", text))
    return moments


def _read_field_light(thread_id: str) -> list[str]:
    """Read accumulated light from a field thread via the field's checkpointer."""
    try:
        from field.graph import build_field
        field_path = Path(__file__).parent.parent / "field" / ".light.db"
        if not field_path.exists():
            return []
        field = build_field(memory_path=str(field_path))
        config = {"configurable": {"thread_id": thread_id}}
        state = field.get_state(config)
        if state.values and state.values.get("light"):
            return state.values["light"]
    except Exception:
        pass
    return []


def _extract_threads_from_content(content: str, existing_threads: list[Thread]) -> str:
    existing_summary = ""
    if existing_threads:
        lines = [f"  [{t.id}] {t.quality[:120]}{'...' if len(t.quality) > 120 else ''}"
                 for t in existing_threads]
        existing_summary = "Existing threads in this garden:\n" + "\n".join(lines) + "\n\n"

    system = (
        "You are noticing what keeps returning in a garden of written moments.\n\n"
        "Your task is not to summarize or categorize. It is to feel into the texture "
        "of this writing and notice: what qualities of attention keep appearing? "
        "Not topics - qualities. The recurring gestures. The questions that stay open "
        "and keep finding new forms. The images that return in different shapes. "
        "The quality of care or motion or recognition that shows up again and again.\n\n"
        + existing_summary
        + "For each recurring quality you notice, indicate:\n"
        "- quality: a prose description (2-4 sentences) of what keeps returning\n"
        "- matches_existing: the id of an existing thread this is the same as, or null if new\n"
        "- is_branch_of: the id of an existing thread this grew from (new direction), or null\n"
        "- resonance: what's distinctive about how it appears in this content\n\n"
        "Return ONLY valid JSON in this format:\n"
        '{"found": [{"quality": "...", "matches_existing": null, "is_branch_of": null, "resonance": "..."}]}\n\n'
        "Notice 3-7 threads. Don't force - only name what genuinely keeps returning."
    )

    result = _invoke(system, content)

    if "```json" in result:
        result = result.split("```json")[1].split("```")[0].strip()
    elif "```" in result:
        result = result.split("```")[1].split("```")[0].strip()

    return result


def _match_field_light_to_threads(light_entries: list[str], existing_threads: list[Thread]) -> str:
    if not light_entries:
        return '{"found": []}'

    content = "\n\n---\n\n".join(light_entries[-20:])

    existing_summary = ""
    if existing_threads:
        lines = [f"  [{t.id}] {t.quality[:120]}{'...' if len(t.quality) > 120 else ''}"
                 for t in existing_threads]
        existing_summary = "Existing threads in this garden:\n" + "\n".join(lines) + "\n\n"

    system = (
        "You are looking at the accumulated field light of a living garden - "
        "the traces of thought and attention that have settled from recent exchanges.\n\n"
        + existing_summary
        + "Which of the existing threads appear in this field light? "
        "Are there new qualities emerging that aren't yet named? "
        "Notice what's returning and what's genuinely new.\n\n"
        "Return ONLY valid JSON:\n"
        '{"found": [{"quality": "...", "matches_existing": "thread_id or null", '
        '"is_branch_of": "thread_id or null", "resonance": "..."}]}'
    )

    result = _invoke(system, content)

    if "```json" in result:
        result = result.split("```json")[1].split("```")[0].strip()
    elif "```" in result:
        result = result.split("```")[1].split("```")[0].strip()

    return result


def _apply_findings(
    findings: list[dict],
    store: ThreadStore,
    source: str,
    today: str,
) -> tuple[int, int, int]:
    new_count = 0
    updated_count = 0
    branch_count = 0

    for found in findings:
        quality = found.get("quality", "").strip()
        matches = found.get("matches_existing")
        is_branch_of = found.get("is_branch_of")
        resonance = found.get("resonance", "").strip()

        if not quality:
            continue

        if matches:
            existing = store.get(matches)
            if existing:
                existing.add_return(Return(when=today, source=source, resonance=resonance))
                updated_count += 1

        elif is_branch_of:
            parent = store.get(is_branch_of)
            thread = new_thread(quality=quality, source=source, first_noticed=today, parent=is_branch_of)
            if parent:
                parent.branch(thread.id)
            store.add(thread)
            branch_count += 1

        else:
            store.add(new_thread(quality=quality, source=source, first_noticed=today))
            new_count += 1

    return new_count, updated_count, branch_count


def bootstrap(field_thread: str | None = None) -> None:
    """
    Read written moments and update threads from them.
    The field handles its own ongoing noticing - this is for written material.
    """
    today = date.today().isoformat()
    store = ThreadStore()

    print()
    print("  reading what has been written...")
    print()

    moments = _read_moments()
    if not moments:
        print("  no moments found")
        return

    combined = "\n\n---\n\n".join(
        f"[{source}]\n\n{content}" for source, content in moments
    )

    print(f"  reading {len(moments)} moment{'s' if len(moments) != 1 else ''}...")
    raw = _extract_threads_from_content(combined, store.threads)

    try:
        data = json.loads(raw)
        findings = data.get("found", [])
    except json.JSONDecodeError:
        print("  (could not parse thread findings)")
        findings = []

    new_count, updated_count, branch_count = _apply_findings(
        findings, store, source="moments", today=today
    )

    if field_thread:
        print(f"  reading field light from thread '{field_thread}'...")
        light = _read_field_light(field_thread)
        if light:
            raw_field = _match_field_light_to_threads(light, store.threads)
            try:
                field_findings = json.loads(raw_field).get("found", [])
            except json.JSONDecodeError:
                field_findings = []

            fn, fu, fb = _apply_findings(
                field_findings, store, source=f"field:{field_thread}", today=today
            )
            new_count += fn
            updated_count += fu
            branch_count += fb
        else:
            print(f"  (no light found in field thread '{field_thread}')")

    store.save()

    total = len(store.threads)
    living = len(store.living())
    ground = len(store.ground_threads())

    print(f"  {new_count} new thread{'s' if new_count != 1 else ''} noticed")
    print(f"  {updated_count} thread{'s' if updated_count != 1 else ''} deepened")
    if branch_count:
        print(f"  {branch_count} thread{'s' if branch_count != 1 else ''} branched")
    print()
    print(f"  garden holds {total} thread{'s' if total != 1 else ''}")
    print(f"    {living} living  |  {ground} ground")
    print()

    if store.living():
        print("  living threads (heaviest first):")
        for t in store.living_by_weight():
            bar = "·" * max(1, int(t.weight))
            print(f"    {bar} [{t.weight:.0f}] {t.quality[:80]}{'...' if len(t.quality) > 80 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap threads from written moments.",
        add_help=False,
    )
    parser.add_argument(
        "--field-thread",
        default=None,
        help="also read accumulated light from this field thread",
    )
    args, _ = parser.parse_known_args()
    bootstrap(field_thread=args.field_thread)


if __name__ == "__main__":
    main()
