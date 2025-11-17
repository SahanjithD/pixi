from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Iterable, List

from pixi.perception import VisionEvent, VisionProcessor
from pixi.reasoning_engine import ReasoningEngine
from pixi.state_manager import StateManager


def _events_to_payload(events: Iterable[VisionEvent]) -> List[dict]:
    payload: List[dict] = []
    for event in events:
        payload.append(
            {
                "summary": event.summary,
                "weight": event.weight,
                "data": event.data,
            }
        )
    return payload


def main(*, display: bool = False, cooldown: float = 0.3) -> None:
    state = StateManager()
    engine = ReasoningEngine(state_manager=state)
    vision = VisionProcessor()

    stop_requested = False

    def _request_stop(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)

    print("[Runtime] Starting realtime loop. Press Ctrl+C or 'q' (window) to exit.")

    for events in vision.stream_events(display=display):
        if stop_requested:
            break

        payload = _events_to_payload(events)
        if not payload:
            # Provide a small idle event so the LLM still receives context.
            payload = [{"summary": "no_event", "weight": 0.05}]

        decision = engine.decide_action(payload)
        action = decision["action"]
        reason = decision["reason"]
        print(f"[Runtime] Action: {action.value} — {reason}")

        time.sleep(cooldown)

    print("[Runtime] Stopping realtime loop.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pixi with real camera input.")
    parser.add_argument("--display", action="store_true", help="Show annotated camera preview.")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.3,
        help="Delay in seconds between reasoning cycles (prevents excessive API calls).",
    )
    args = parser.parse_args()

    try:
        main(display=args.display, cooldown=args.cooldown)
    except RuntimeError as exc:
        print(f"[Runtime] Error: {exc}")
        sys.exit(1)
