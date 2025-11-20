from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

from pixi.vision.perception import VisionProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the vision pipeline and print detected events."
    )
    parser.add_argument("--display", action="store_true", help="Show annotated camera preview.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index to open.",
    )
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
        default=5,
        help="Frames to keep tracking before re-running face detection.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to wait after each frame batch when printing events.",
    )
    return parser.parse_args()


def main(
    *,
    display: bool = False,
    camera_index: int = 0,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    hand_interval: int = 5,
    enable_hands: bool = True,
    enable_face_tracking: bool = True,
    face_smoothing: float = 0.6,
    tracking_timeout: float = 1.2,
    face_interval: int = 10,
    sleep_seconds: float = 0.2,
) -> None:
    processor = VisionProcessor(
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

    print("[Vision] Starting vision-only loop. Press Ctrl+C or 'q' (window) to exit.")

    try:
        for events in processor.stream_events(display=display):
            if stop_requested:
                break

            if not events:
                print("[Vision] No events detected.")
            else:
                for event in events:
                    print(
                        f"[Vision] {event.summary} (weight={event.weight:.2f}) -> {event.data}"
                    )

            time.sleep(max(0.01, sleep_seconds))
    except RuntimeError as exc:
        print(f"[Vision] Error: {exc}")
        raise
    finally:
        print("[Vision] Loop stopped.")


def cli() -> None:
    args = parse_args()
    try:
        main(
            display=args.display,
            camera_index=args.camera_index,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            hand_interval=args.hand_interval,
            enable_hands=args.enable_hands,
            enable_face_tracking=args.enable_face_tracking,
            face_smoothing=args.face_smoothing,
            tracking_timeout=args.tracking_timeout,
            face_interval=args.face_interval,
            sleep_seconds=args.sleep,
        )
    except RuntimeError:
        sys.exit(1)


if __name__ == "__main__":
    cli()
