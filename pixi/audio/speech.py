from __future__ import annotations

import json
import math
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional import guard
    from vosk import KaldiRecognizer, Model
except (ImportError, ModuleNotFoundError):  # pragma: no cover - import guard
    Model = None  # type: ignore[assignment]
    KaldiRecognizer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional import guard
    from pvrecorder import PvRecorder
except (ImportError, ModuleNotFoundError):  # pragma: no cover - import guard
    PvRecorder = None  # type: ignore[assignment]


class SpeechCaptureError(RuntimeError):
    """Raised when an audio capture or transcription step fails."""


@dataclass(slots=True)
class SpeechCaptureConfig:
    """Configuration for recording speech after a wake word."""

    device_index: int = -1
    sample_rate: int = 16000
    frame_length: int = 512
    max_duration: float = 4.5
    min_duration: float = 0.35
    silence_timeout: float = 0.8
    silence_threshold: float = 550.0


class SpeechRecorder:
    """Utility that records PCM audio using PvRecorder until silence or max duration."""

    def __init__(self, config: SpeechCaptureConfig) -> None:
        if PvRecorder is None:  # pragma: no cover - import guard
            raise SpeechCaptureError(
                "Speech recording requires `pvrecorder`. Install it with `pip install pvrecorder`."
            )
        self._config = config

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    def capture(self) -> bytes:
        cfg = self._config
        if cfg.sample_rate not in (16000, 0):
            # PvRecorder operates at 16 kHz; surface mismatch without failing hard.
            raise SpeechCaptureError("PvRecorder captures at 16 kHz; adjust SpeechCaptureConfig.sample_rate accordingly.")
        recorder = PvRecorder(device_index=cfg.device_index, frame_length=cfg.frame_length)
        frames: list[array] = []
        start_time = time.time()
        silence_start: Optional[float] = None

        try:
            recorder.start()
            while True:
                pcm = recorder.read()
                if pcm is None:
                    continue
                frame = array("h", pcm)
                frames.append(frame)

                elapsed = time.time() - start_time
                if elapsed >= cfg.max_duration:
                    break

                peak = max(abs(sample) for sample in pcm)
                energy = math.sqrt(sum(sample * sample for sample in pcm) / len(pcm))

                # Reset silence tracker when energy crosses threshold.
                if peak >= cfg.silence_threshold or energy >= cfg.silence_threshold:
                    silence_start = None
                else:
                    if elapsed >= cfg.min_duration:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= cfg.silence_timeout:
                            break
        finally:
            try:
                recorder.stop()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            recorder.delete()

        if not frames:
            raise SpeechCaptureError("No audio captured after wake word.")

        buffer = array("h")
        for frame in frames:
            buffer.extend(frame)
        return buffer.tobytes()


class VoskSpeechRecognizer:
    """Thin wrapper around a Vosk model to transcribe PCM16 audio buffers."""

    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        if Model is None or KaldiRecognizer is None:  # pragma: no cover - import guard
            raise SpeechCaptureError("Transcription requires the `vosk` package. Install it with `pip install vosk`.")

        resolved_path = self._resolve_model_dir(model_path)
        if resolved_path is None:
            raise SpeechCaptureError(
                f"Unable to locate a Vosk model in '{model_path}'. Ensure the directory contains 'conf/model.conf'."
            )

        try:
            self._model = Model(str(resolved_path))
        except Exception as exc:  # pragma: no cover - library specific error
            raise SpeechCaptureError(f"Failed to load Vosk model from '{resolved_path}': {exc}") from exc

        self._sample_rate = sample_rate

    @staticmethod
    def _resolve_model_dir(path: Path) -> Optional[Path]:
        if (path / "conf" / "model.conf").exists():
            return path
        nested = list(path.glob("**/conf/model.conf"))
        if not nested:
            return None
        # Prefer the shallowest path to avoid nested duplicates.
        nested.sort(key=lambda p: len(p.parts))
        return nested[0].parent.parent

    def transcribe(self, pcm_bytes: bytes) -> dict:
        recognizer = KaldiRecognizer(self._model, self._sample_rate)
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(pcm_bytes)
        result = recognizer.FinalResult()
        try:
            data = json.loads(result)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise SpeechCaptureError(f"Vosk returned invalid JSON: {exc}") from exc
        return data

    def transcribe_text(self, pcm_bytes: bytes) -> str:
        data = self.transcribe(pcm_bytes)
        return data.get("text", "").strip()