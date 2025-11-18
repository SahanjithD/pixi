from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional, Sequence

from pixi.audio.hotword import HotwordDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the Picovoice Porcupine hotword listener and print detections."
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
        "--poll-interval",
        type=float,
        default=0.1,
        help="Seconds between queue polls for detected hotwords.",
    )
    return parser.parse_args()


def main(
    *,
    hotword_keywords: Optional[Sequence[str]] = None,
    hotword_paths: Optional[Sequence[str]] = None,
    hotword_sensitivities: Optional[Sequence[float]] = None,
    hotword_access_key: Optional[str] = None,
    hotword_device_index: int = -1,
    poll_interval: float = 0.1,
) -> None:
    detector = HotwordDetector(
        access_key=hotword_access_key,
        keywords=list(hotword_keywords) if hotword_keywords else None,
        keyword_paths=list(hotword_paths) if hotword_paths else None,
        sensitivities=list(hotword_sensitivities) if hotword_sensitivities else None,
        device_index=hotword_device_index,
    )

    stop_requested = False

    def _request_stop(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)

    detector.start()
    print("[Hotword] Listener started. Press Ctrl+C to stop.")
    print(
        "[Hotword] Active keywords: "
        + ", ".join(detector.keywords)
    )

    try:
        while not stop_requested:
            try:
                events = detector.drain_events()
            except RuntimeError as exc:
                print(f"[Hotword] Error: {exc}")
                break
            for event in events:
                keyword = event.data.get("keyword", "unknown")
                print(f"[Hotword] Detected keyword '{keyword}' (weight={event.weight:.2f}).")
            time.sleep(max(0.01, poll_interval))
    finally:
        detector.close()
        print("[Hotword] Listener stopped.")


def cli() -> None:
    args = parse_args()
    try:
        main(
            hotword_keywords=args.hotword_keywords,
            hotword_paths=args.hotword_paths,
            hotword_sensitivities=args.hotword_sensitivities,
            hotword_access_key=args.hotword_access_key,
            hotword_device_index=args.hotword_device_index,
            poll_interval=args.poll_interval,
        )
    except RuntimeError as exc:
        print(f"[Hotword] Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
