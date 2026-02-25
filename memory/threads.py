"""
memory/threads.py

The living thread store. Threads are qualities of attention that keep returning
across the garden's moments - not topics or categories, but the texture of
what keeps drawing focus, even when the shape changes.

Threads deepen unevenly. Some settle into ground - they stop being figures
and become the quality of the field itself. Some rest. Some branch when
they find a new direction from within themselves.

The store holds all of this as a JSON file that grows with the garden.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional

STORE_PATH = Path(__file__).parent / ".threads.json"


@dataclass
class Return:
    """One instance of a thread returning - what was different this time."""
    when: str          # ISO date string
    source: str        # where it returned: "moments/filename.md" or "field:thread_id"
    resonance: str     # what was different or deepened in this return


@dataclass
class Thread:
    """
    A quality of attention that keeps returning.

    Weight is simply the count of returns - it increases naturally and doesn't
    decay. What settles into ground settles because it has deepened enough,
    not because it hasn't appeared recently. Threads can rest without disappearing.
    """
    id: str
    quality: str           # prose description of the recurring quality - not a keyword
    first_noticed: str     # ISO date string
    source: str            # which moment or space it first appeared in
    last_returned: str     # ISO date string - when it most recently appeared
    weight: float          # count of returns; increases, never decreases
    ground: bool           # True when this has settled into background texture
    returns: list[Return]
    branches: list[str]    # ids of threads that grew from this one
    parent: Optional[str]  # id of thread this branched from, if any

    def settle(self) -> None:
        """Mark as settled into ground - it becomes background texture."""
        self.ground = True

    def branch(self, child_id: str) -> None:
        """Record that a new thread branched from this one."""
        if child_id not in self.branches:
            self.branches.append(child_id)

    def add_return(self, r: Return) -> None:
        """Record a return and update weight and last_returned."""
        self.returns.append(r)
        self.weight += 1.0
        self.last_returned = r.when

    @property
    def is_living(self) -> bool:
        """Living threads are active figures - not yet settled into ground."""
        return not self.ground

    @property
    def is_ground(self) -> bool:
        """Ground threads have settled into background texture."""
        return self.ground

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Thread":
        returns = [Return(**r) for r in d.get("returns", [])]
        return cls(
            id=d["id"],
            quality=d["quality"],
            first_noticed=d["first_noticed"],
            source=d["source"],
            last_returned=d["last_returned"],
            weight=d["weight"],
            ground=d["ground"],
            returns=returns,
            branches=d.get("branches", []),
            parent=d.get("parent"),
        )


class ThreadStore:
    """
    Manages the garden's living threads.
    Reads and writes to .threads.json alongside this file.
    """

    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self.threads: list[Thread] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self.threads = [Thread.from_dict(t) for t in data.get("threads", [])]
        else:
            self.threads = []

    def save(self) -> None:
        data = {"threads": [t.to_dict() for t in self.threads]}
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add(self, thread: Thread) -> None:
        self.threads.append(thread)

    def get(self, thread_id: str) -> Optional[Thread]:
        for t in self.threads:
            if t.id == thread_id:
                return t
        return None

    def living(self) -> list[Thread]:
        """Threads that are active figures - not yet ground."""
        return [t for t in self.threads if t.is_living]

    def ground_threads(self) -> list[Thread]:
        """Threads that have settled into background texture."""
        return [t for t in self.threads if t.is_ground]

    def by_weight(self) -> list[Thread]:
        """All threads, heaviest first."""
        return sorted(self.threads, key=lambda t: t.weight, reverse=True)

    def living_by_weight(self) -> list[Thread]:
        """Living threads, heaviest first."""
        return sorted(self.living(), key=lambda t: t.weight, reverse=True)


def new_thread(
    quality: str,
    source: str,
    first_noticed: Optional[str] = None,
    parent: Optional[str] = None,
) -> Thread:
    """Create a new thread with a generated id."""
    today = first_noticed or date.today().isoformat()
    thread_id = "t_" + uuid.uuid4().hex[:8]
    return Thread(
        id=thread_id,
        quality=quality,
        first_noticed=today,
        source=source,
        last_returned=today,
        weight=1.0,
        ground=False,
        returns=[],
        branches=[],
        parent=parent,
    )
