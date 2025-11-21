from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
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
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="After a hotword fires, capture a short utterance and transcribe it with Vosk.",
    )
    parser.add_argument(
        "--vosk-model-path",
        help="Path to a Vosk model directory. Default: ./vosk-model-small-en-us-0.15.",
    )
    parser.add_argument(
        "--capture-duration",
        type=float,
        default=4.0,
        help="Maximum seconds of speech to record after the wake word.",
    )
    parser.add_argument(
        "--min-capture-duration",
        type=float,
        default=0.35,
        help="Minimum seconds to record before allowing silence-based stop.",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.8,
        help="Seconds of low energy required to stop recording after the minimum duration.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=550.0,
        help="Amplitude/RMS threshold to determine silence for end-of-speech detection.",
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
    transcribe: bool = False,
    vosk_model_path: Optional[str] = None,
    capture_duration: float = 4.0,
    min_capture_duration: float = 0.35,
    silence_duration: float = 0.8,
    silence_threshold: float = 550.0,
) -> None:
    keyword_list = list(hotword_keywords) if hotword_keywords else None
    path_list = list(hotword_paths) if hotword_paths else None
    sensitivity_list = list(hotword_sensitivities) if hotword_sensitivities else None

    def _create_detector() -> HotwordDetector:
        return HotwordDetector(
            access_key=hotword_access_key,
            keywords=list(keyword_list) if keyword_list else None,
            keyword_paths=list(path_list) if path_list else None,
            sensitivities=list(sensitivity_list) if sensitivity_list else None,
            device_index=hotword_device_index,
        )

    speech_recorder = None
    speech_transcriber = None
    if transcribe:
        try:
            from pixi.audio.speech import (
                SpeechCaptureConfig,
                SpeechCaptureError,
                SpeechRecorder,
                VoskSpeechRecognizer,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Transcription requires optional dependencies. Ensure `vosk` and `pvrecorder` are installed."
            ) from exc

        default_model_dir = Path(__file__).resolve().parent.parent.parent / "vosk-model-small-en-us-0.15"
        model_dir = Path(vosk_model_path) if vosk_model_path else default_model_dir
        capture_config = SpeechCaptureConfig(
            device_index=hotword_device_index,
            max_duration=max(0.5, capture_duration),
            min_duration=max(0.05, min_capture_duration),
            silence_timeout=max(0.1, silence_duration),
            silence_threshold=max(50.0, silence_threshold),
        )
        speech_recorder = SpeechRecorder(capture_config)
        speech_transcriber = VoskSpeechRecognizer(model_dir, sample_rate=capture_config.sample_rate)

    stop_requested = False

    def _request_stop(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)

    detector = _create_detector()
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
                if transcribe and speech_recorder and speech_transcriber:
                    detector.close()
                    detector = None
                    try:
                        pcm = speech_recorder.capture()
                    except SpeechCaptureError as exc:
                        print(f"[Hotword] Speech capture failed: {exc}")
                        if stop_requested:
                            break
                        detector = _create_detector()
                        detector.start()
                        continue
                    try:
                        transcript_data = speech_transcriber.transcribe(pcm)
                    except SpeechCaptureError as exc:
                        print(f"[Hotword] Transcription failed: {exc}")
                        if stop_requested:
                            break
                    else:
                        transcript_text = transcript_data.get("text", "").strip()
                        print("[Hotword] Transcript:", transcript_text if transcript_text else "<no speech>")
                        if transcript_data.get("result"):
                            words = " ".join(word.get("word", "") for word in transcript_data["result"])
                            if words and words != transcript_text:
                                print(f"[Hotword]   Words: {words}")
                    if stop_requested:
                        break
                    detector = _create_detector()
                    detector.start()
            time.sleep(max(0.01, poll_interval))
    finally:
        if detector:
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
            transcribe=args.transcribe,
            vosk_model_path=args.vosk_model_path,
            capture_duration=args.capture_duration,
            min_capture_duration=args.min_capture_duration,
            silence_duration=args.silence_duration,
            silence_threshold=args.silence_threshold,
        )
    except RuntimeError as exc:
        print(f"[Hotword] Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
