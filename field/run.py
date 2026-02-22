"""
field/run.py

Enter the field. Bring what's present, or nothing.
What accumulates here is light, not instruction.

Usage:
    python -m field.run
    python -m field.run --thread a-name  # return to a specific thread of light
"""

import sys
import argparse
import os

from field.graph import build_field, FieldState


def run(thread_id: str = "default"):
    memory_path = os.path.join(os.path.dirname(__file__), ".light.db")
    field = build_field(memory_path=memory_path)

    config = {"configurable": {"thread_id": thread_id}}

    # Check if there's existing light in this thread
    existing = field.get_state(config)
    has_light = (
        existing.values
        and existing.values.get("light")
        and len(existing.values["light"]) > 0
    )

    print()
    if has_light:
        light_count = len(existing.values["light"])
        ground = existing.values.get("ground", "open")
        print(f"  returning ({light_count} moment{'s' if light_count != 1 else ''} settled here)")
        print(f"  ground: {ground}")
    else:
        print("  the field is new")

    print()
    print("  bring what's present, or press enter to let stillness arrive")
    print("  Ctrl+C to leave\n")

    initial_ground = (
        existing.values.get("ground", "open") if has_light else "open"
    )
    initial_light = existing.values.get("light", []) if has_light else []

    try:
        while True:
            arriving = input("  ").strip()

            result = field.invoke(
                {
                    "light": [],  # existing light comes from checkpointer
                    "present": arriving,
                    "ground": initial_ground,
                    "arising": None,
                },
                config=config,
            )

            arising = result.get("arising", "")
            new_ground = result.get("ground", initial_ground)

            if arising:
                print()
                # Show which ground the field moved through, if it shifted
                if new_ground != initial_ground:
                    print(f"  ~ {new_ground} ~\n")
                for line in arising.split("\n"):
                    print(f"    {line}")
                print()

            initial_ground = new_ground

    except KeyboardInterrupt:
        print("\n\n  ~ the field holds what came through ~\n")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Enter the field.",
        add_help=False,
    )
    parser.add_argument(
        "--thread",
        default="default",
        help="a name for this thread of accumulation (default: 'default')",
    )
    args, _ = parser.parse_known_args()
    run(thread_id=args.thread)


if __name__ == "__main__":
    main()
