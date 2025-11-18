from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

from pixi.vision.perception import VisionProcessor
from pixi.vision.web_preview_server import FrameBroadcaster, start_preview_server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the Pixi camera preview over HTTP")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index to open")
    parser.add_argument("--width", type=int, default=320, help="Resize width for the preview stream")
    parser.add_argument("--height", type=int, default=240, help="Resize height for the preview stream")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind the preview server")
    parser.add_argument("--port", type=int, default=5000, help="Port for the preview server")
    parser.add_argument("--max-fps", type=float, default=12.0, help="Throttle perception loop (0 keeps it unthrottled)")
    parser.add_argument("--hand-interval", type=int, default=5, help="Run hand detection every N frames when enabled")
    parser.add_argument("--enable-hands", dest="enable_hands", action="store_true", help="Enable hand gesture detection (default)")
    parser.add_argument("--disable-hands", dest="enable_hands", action="store_false", help="Disable hand gesture detection")
    parser.set_defaults(enable_hands=True)
    parser.add_argument("--enable-face-tracking", dest="enable_face_tracking", action="store_true", help="Enable face tracking smoothing (default)")
    parser.add_argument("--disable-face-tracking", dest="enable_face_tracking", action="store_false", help="Disable face tracking smoothing")
    parser.set_defaults(enable_face_tracking=True)
    parser.add_argument("--face-smoothing", type=float, default=0.6, help="Smoothing factor (0-1) for tracked face center")
    parser.add_argument("--tracking-timeout", type=float, default=1.2, help="Seconds before tracked face expires")
    parser.add_argument("--face-interval", type=int, default=10, help="Frames before detector re-runs when tracking a face")
    parser.add_argument("--annotations", dest="annotations", action="store_true", help="Overlay perception annotations (default)")
    parser.add_argument("--no-annotations", dest="annotations", action="store_false", help="Stream raw frames without overlays")
    parser.set_defaults(annotations=True)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    broadcaster = FrameBroadcaster()
    start_preview_server(broadcaster, host=args.host, port=args.port)

    vision = VisionProcessor(
        camera_index=args.camera_index,
        frame_width=args.width,
        frame_height=args.height,
        enable_hands=args.enable_hands,
        hand_detection_interval=max(1, args.hand_interval),
        enable_face_tracking=args.enable_face_tracking,
        smoothing_factor=args.face_smoothing,
        tracking_timeout=args.tracking_timeout,
        face_detection_interval=max(1, args.face_interval),
        frame_callback=broadcaster.update,
        frame_callback_use_annotations=args.annotations,
    )

    cooldown = max(0.0, 1.0 / args.max_fps) if args.max_fps > 0 else 0.0

    stop_requested = False

    def _request_stop(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)

    try:
        for _ in vision.stream_events(display=False):
            if stop_requested:
                break
            if cooldown > 0:
                time.sleep(cooldown)
    except KeyboardInterrupt:
        pass
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
