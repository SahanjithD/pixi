from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Iterable, List, Optional, Sequence, Union

from pixi.audio.hotword import AudioEvent, HotwordDetector
from pixi.core.reasoning_engine import ReasoningEngine
from pixi.core.state_manager import StateManager
from pixi.vision.perception import VisionEvent, VisionProcessor
from pixi.vision.web_preview_server import FrameBroadcaster, start_preview_server


PerceptionEvent = Union[VisionEvent, AudioEvent]


def _events_to_payload(events: Iterable[PerceptionEvent]) -> List[dict]:
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
    enable_vision: bool = True,
    hand_interval: int = 5,
    enable_hands: bool = True,
    enable_face_tracking: bool = True,
    face_smoothing: float = 0.6,
    tracking_timeout: float = 1.2,
    face_interval: int = 10,
    enable_hotword: bool = False,
    hotword_keywords: Optional[Sequence[str]] = None,
    hotword_paths: Optional[Sequence[str]] = None,
    hotword_sensitivities: Optional[Sequence[float]] = None,
    hotword_access_key: Optional[str] = None,
    hotword_device_index: int = -1,
    web_preview: bool = False,
    preview_host: str = "0.0.0.0",
    preview_port: int = 5000,
    preview_annotations: bool = True,
    preview_backend: str = "browser",
    # New: event-driven decision controls
    event_driven: bool = True,
    min_interval: float = 0.15,
    max_interval: float = 2.0,
    move_threshold: float = 0.05,
    area_threshold: float = 0.12,
) -> None:
    state = StateManager()
    engine = ReasoningEngine(state_manager=state)
    frame_callback = None
    frame_callback_use_annotations = True

    display_effective = display
    if web_preview and enable_vision:
        if preview_backend == "browser":
            broadcaster = FrameBroadcaster()
            start_preview_server(
                broadcaster,
                host=preview_host,
                port=preview_port,
                title="Pixi Runtime Preview",
            )
            frame_callback = broadcaster.update
            frame_callback_use_annotations = preview_annotations
        elif preview_backend == "opencv":
            display_effective = True
            print("[Runtime] Web preview set to OpenCV window; starting annotated display.")
        else:
            raise ValueError(f"Unsupported preview backend: {preview_backend}")
    elif web_preview:
        print("[Runtime] Preview requested but vision is disabled; no frames will be served.")
    elif preview_backend != "browser":
        print("[Runtime] Ignoring --preview-backend because --web-preview was not provided.")

    vision: Optional[VisionProcessor] = None
    vision_stream = None
    if enable_vision:
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
            frame_callback=frame_callback,
            frame_callback_use_annotations=frame_callback_use_annotations,
        )
        vision_stream = vision.stream_events(display=display_effective)

    stop_requested = False
    # Track last decision context for event-driven triggering
    last_decision_time = time.time()
    last_face: Optional[tuple[float, float, float]] = None  # (cx, cy, area)
    last_gesture: Optional[str] = None
    last_hotword: Optional[str] = None

    def _request_stop(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    hotword_detector: Optional[HotwordDetector] = None
    if enable_hotword:
        try:
            hotword_detector = HotwordDetector(
                access_key=hotword_access_key,
                keywords=list(hotword_keywords) if hotword_keywords else None,
                keyword_paths=list(hotword_paths) if hotword_paths else None,
                sensitivities=list(hotword_sensitivities) if hotword_sensitivities else None,
                device_index=hotword_device_index,
            )
            hotword_detector.start()
            print(
                "[Hotword] Listening for keywords: "
                f"{', '.join(hotword_detector.keywords)}"
            )
        except Exception as exc:
            print(f"[Hotword] Failed to start Porcupine: {exc}")
            if hotword_detector is not None:
                hotword_detector.close()
            return

    signal.signal(signal.SIGINT, _request_stop)

    print("[Runtime] Starting realtime loop. Press Ctrl+C or 'q' (window) to exit.")

    try:
        while True:
            if stop_requested:
                break

            vision_events: List[VisionEvent] = []
            if vision_stream is not None:
                try:
                    vision_events = next(vision_stream)
                except StopIteration:
                    break

            # Update state from primary face if tracking is enabled
            primary_face = None  # (cx, cy, area, confidence)
            if enable_face_tracking and enable_vision:
                for event in vision_events:
                    if event.summary == "face_detected" and event.data.get("is_primary", False):
                        center_x = float(event.data.get("smoothed_center_x", event.data.get("center_x", 0.5)))
                        center_y = float(event.data.get("smoothed_center_y", event.data.get("center_y", 0.5)))
                        area = float(event.data.get("smoothed_area", event.data.get("area", 0.0)))
                        confidence = float(event.data.get("tracking_confidence", event.data.get("confidence", event.weight)))
                        tracking_id = event.data.get("tracking_id")
                        face_id = f"face_{int(tracking_id)}" if isinstance(tracking_id, (int, float)) else "face_primary"
                        state.update_face_target(face_id=face_id, center_x=center_x, center_y=center_y, area=area, confidence=confidence)
                        primary_face = (center_x, center_y, area, confidence)

            combined_events: List[PerceptionEvent] = list(vision_events)
            # Detect gesture presence/name for triggering
            current_gesture: Optional[str] = None
            for e in vision_events:
                if e.summary == "hand_gesture":
                    current_gesture = str(e.data.get("gesture", "unknown"))
                    break

            # Hotword events (if enabled)
            audio_events: List[AudioEvent] = []
            current_hotword: Optional[str] = None
            if hotword_detector is not None:
                try:
                    audio_events = hotword_detector.drain_events()
                except RuntimeError as exc:
                    print(f"[Hotword] Error: {exc}")
                    break
                if audio_events:
                    for audio_event in audio_events:
                        keyword = audio_event.data.get("keyword", "unknown")
                        print(f"[Hotword] Detected keyword '{keyword}'.")
                        current_hotword = keyword
                    combined_events.extend(audio_events)

            # Decide when to call the reasoning engine
            now = time.time()
            should_decide = False

            if not event_driven:
                # Original polling mode: every cooldown
                should_decide = (now - last_decision_time) >= cooldown
            else:
                # Event-driven triggers
                face_trigger = False
                if primary_face is not None:
                    cx, cy, area, _conf = primary_face
                    if last_face is None:
                        face_trigger = True  # face appeared
                    else:
                        lx, ly, la = last_face
                        moved = (abs(cx - lx) >= move_threshold) or (abs(cy - ly) >= move_threshold)
                        area_changed = (la == 0.0 and area > 0.0) or (la > 0.0 and abs(area - la) / la >= area_threshold)
                        face_trigger = moved or area_changed
                elif last_face is not None:
                    face_trigger = True  # face disappeared

                gesture_trigger = (current_gesture is not None and current_gesture != last_gesture)
                hotword_trigger = (current_hotword is not None)

                heartbeat = (now - last_decision_time) >= max_interval
                # Respect minimum spacing between decisions
                if (face_trigger or gesture_trigger or hotword_trigger or heartbeat) and (now - last_decision_time) >= min_interval:
                    should_decide = True

            if should_decide:
                payload = _events_to_payload(combined_events)
                if not payload:
                    payload = [{"summary": "no_event", "weight": 0.05}]
                decision = engine.decide_action(payload)
                action = decision["action"]
                reason = decision["reason"]
                print(f"[Runtime] Action: {action.value} — {reason}")

                # Update last decision context
                last_decision_time = now
                last_gesture = current_gesture
                last_hotword = current_hotword
                last_face = (primary_face[0], primary_face[1], primary_face[2]) if primary_face else None

            # Sleep briefly; in polling mode we already throttle via cooldown above
            if not event_driven:
                time.sleep(max(0.0, cooldown - max(0.0, time.time() - now)))
            else:
                time.sleep(0.01)
    finally:
        if hotword_detector is not None:
            hotword_detector.close()

    print("[Runtime] Stopping realtime loop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pixi with real camera input.")
    parser.add_argument("--display", action="store_true", help="Show annotated camera preview.")
    parser.add_argument(
        "--disable-vision",
        dest="enable_vision",
        action="store_false",
        help="Disable the vision pipeline (camera + perception).",
    )
    parser.add_argument(
        "--enable-vision",
        dest="enable_vision",
        action="store_true",
        help="Enable the vision pipeline (default).",
    )
    parser.set_defaults(enable_vision=True)
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
        default=5,
        help="Frames to keep tracking before re-running face detection.",
    )
    parser.add_argument(
        "--enable-hotword",
        action="store_true",
        help="Enable Picovoice Porcupine hotword detection.",
    )
    parser.add_argument(
        "--hotword-keyword",
        dest="hotword_keywords",
        action="append",
        help="Built-in Porcupine keyword to monitor (can be repeated). Default: picovoice.",
    )
    parser.add_argument(
        "--hotword-file",
        dest="hotword_paths",
        action="append",
        help="Path to a custom Porcupine .ppn keyword file (can be repeated).",
    )
    parser.add_argument(
        "--hotword-sensitivity",
        dest="hotword_sensitivities",
        action="append",
        type=float,
        help="Sensitivity override (0.0-1.0) per keyword. Provide one per keyword/file.",
    )
    parser.add_argument(
        "--hotword-device-index",
        type=int,
        default=-1,
        help="Audio device index for hotword detection (see `pvrecorder --show_audio_devices`).",
    )
    parser.add_argument(
        "--hotword-access-key",
        help="Override PICOVOICE_ACCESS_KEY environment variable.",
    )
    parser.add_argument(
        "--web-preview",
        action="store_true",
        help="Expose a live preview (paired with --preview-backend).",
    )
    parser.add_argument(
        "--preview-backend",
        choices=("browser", "opencv"),
        default="browser",
        help="Select how the preview is rendered when --web-preview is set.",
    )
    parser.add_argument(
        "--preview-host",
        default="0.0.0.0",
        help="Host/IP for the web preview server (use with --web-preview).",
    )
    parser.add_argument(
        "--preview-port",
        type=int,
        default=5000,
        help="Port for the web preview server (use with --web-preview).",
    )
    parser.add_argument(
        "--preview-annotations",
        dest="preview_annotations",
        action="store_true",
        help="Overlay perception annotations in the web preview (default).",
    )
    parser.add_argument(
        "--preview-no-annotations",
        dest="preview_annotations",
        action="store_false",
        help="Stream raw frames in the web preview without overlays.",
    )
    parser.set_defaults(preview_annotations=True)

    # New CLI for event-driven decisions
    parser.add_argument("--event-driven", dest="event_driven", action="store_true", help="Only decide when perceptual state changes (default).")
    parser.add_argument("--polling", dest="event_driven", action="store_false", help="Decide every --cooldown seconds regardless of changes.")
    parser.set_defaults(event_driven=True)
    parser.add_argument("--min-interval", type=float, default=0.15, help="Minimum seconds between decisions in event-driven mode.")
    parser.add_argument("--max-interval", type=float, default=2.0, help="Maximum seconds between decisions (heartbeat) in event-driven mode.")
    parser.add_argument("--move-threshold", type=float, default=0.05, help="Face move threshold (normalized) to trigger a decision.")
    parser.add_argument("--area-threshold", type=float, default=0.12, help="Relative face area delta to trigger a decision (e.g., 0.12 = 12%).")
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    main(
        display=args.display,
        cooldown=args.cooldown,
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        enable_vision=args.enable_vision,
        hand_interval=args.hand_interval,
        enable_hands=args.enable_hands,
        enable_face_tracking=args.enable_face_tracking,
        face_smoothing=args.face_smoothing,
        tracking_timeout=args.tracking_timeout,
        face_interval=args.face_interval,
        enable_hotword=args.enable_hotword,
        hotword_keywords=args.hotword_keywords,
        hotword_paths=args.hotword_paths,
        hotword_sensitivities=args.hotword_sensitivities,
        hotword_access_key=args.hotword_access_key,
        hotword_device_index=args.hotword_device_index,
        web_preview=args.web_preview,
        preview_host=args.preview_host,
        preview_port=args.preview_port,
        preview_annotations=args.preview_annotations,
        preview_backend=args.preview_backend,
        event_driven=args.event_driven,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        move_threshold=args.move_threshold,
        area_threshold=args.area_threshold,
    )


if __name__ == "__main__":
    try:
        cli()
    except RuntimeError as exc:
        print(f"[Runtime] Error: {exc}")
        sys.exit(1)
