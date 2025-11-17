from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Iterable, List, Optional

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


def main(
    *,
    display: bool = False,
    cooldown: float = 0.3,
    camera_index: int = 0,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    hand_interval: int = 5,
    enable_hands: bool = True,
    enable_face_tracking: bool = True,
    face_smoothing: float = 0.6,
    tracking_timeout: float = 1.2,
    face_interval: int = 10,
) -> None:
    state = StateManager()
    engine = ReasoningEngine(state_manager=state)
    vision = VisionProcessor(
        camera_index=camera_index,
        frame_width=frame_width,
        frame_height=frame_height,
        enable_hands=enable_hands,
        hand_detection_interval=hand_interval,
        enable_face_tracking=enable_face_tracking,
        smoothing_factor=face_smoothing,
        tracking_timeout=tracking_timeout,
        face_detection_interval=face_interval,
    )

    stop_requested = False

    def _request_stop(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)

    print("[Runtime] Starting realtime loop. Press Ctrl+C or 'q' (window) to exit.")

    for events in vision.stream_events(display=display):
        if stop_requested:
            break

        if enable_face_tracking:
            for event in events:
                if event.summary == "face_detected" and event.data.get("is_primary", False):
                    tracking_id = event.data.get("tracking_id")
                    face_id = (
                        f"face_{int(tracking_id)}"
                        if isinstance(tracking_id, (int, float))
                        else "face_primary"
                    )
                    center_x = float(
                        event.data.get("smoothed_center_x", event.data.get("center_x", 0.5))
                    )
                    center_y = float(
                        event.data.get("smoothed_center_y", event.data.get("center_y", 0.5))
                    )
                    area = float(event.data.get("smoothed_area", event.data.get("area", 0.0)))
                    confidence = float(
                        event.data.get("tracking_confidence", event.data.get("confidence", event.weight))
                    )
                    state.update_face_target(
                        face_id=face_id,
                        center_x=center_x,
                        center_y=center_y,
                        area=area,
                        confidence=confidence,
                    )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pixi with real camera input.")
    parser.add_argument("--display", action="store_true", help="Show annotated camera preview.")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.3,
        help="Delay in seconds between reasoning cycles (prevents excessive API calls).",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index to open.")
    parser.add_argument(
        "--frame-width",
        type=int,
        help="Resize frames to this width before processing (preserves aspect ratio if only width).",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        help="Resize frames to this height before processing (preserves aspect ratio if only height).",
    )
    parser.add_argument(
        "--hand-interval",
        type=int,
        default=5,
        help="Run hand detection every N frames to save compute.",
    )
    parser.add_argument(
        "--enable-hands",
        dest="enable_hands",
        action="store_true",
        help="Enable hand gesture detection (default).",
    )
    parser.add_argument(
        "--disable-hands",
        dest="enable_hands",
        action="store_false",
        help="Disable hand gesture detection entirely.",
    )
    parser.set_defaults(enable_hands=True)
    parser.add_argument(
        "--enable-face-tracking",
        dest="enable_face_tracking",
        action="store_true",
        help="Enable face tracking smoothing and IDs (default).",
    )
    parser.add_argument(
        "--disable-face-tracking",
        dest="enable_face_tracking",
        action="store_false",
        help="Disable face tracking metadata updates.",
    )
    parser.set_defaults(enable_face_tracking=True)
    parser.add_argument(
        "--face-smoothing",
        type=float,
        default=0.6,
        help="Smoothing factor (0-1) for face tracking (higher favors stability).",
    )
    parser.add_argument(
        "--tracking-timeout",
        type=float,
        default=1.2,
        help="Seconds before a tracked face expires without detections.",
    )
    parser.add_argument(
        "--face-interval",
        type=int,
        default=10,
        help="Frames to keep tracking before re-running face detection.",
    )
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    main(
        display=args.display,
        cooldown=args.cooldown,
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        hand_interval=args.hand_interval,
        enable_hands=args.enable_hands,
        enable_face_tracking=args.enable_face_tracking,
        face_smoothing=args.face_smoothing,
        tracking_timeout=args.tracking_timeout,
        face_interval=args.face_interval,
    )


if __name__ == "__main__":
    try:
        cli()
    except RuntimeError as exc:
        print(f"[Runtime] Error: {exc}")
        sys.exit(1)
