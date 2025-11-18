from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv


try:
    import pvporcupine
    from pvrecorder import PvRecorder
except (ImportError, ModuleNotFoundError):  # pragma: no cover - import guard
    pvporcupine = None  # type: ignore[assignment]
    PvRecorder = None  # type: ignore[assignment]


@dataclass(slots=True)
class AudioEvent:
    """Structured description of an audio-derived perception event."""

    summary: str
    weight: float
    data: Dict[str, float | str | bool | Dict[str, float | str | bool]]


class HotwordDetector:
    """Background hotword listener powered by Picovoice Porcupine."""

    def __init__(
        self,
        *,
        access_key: Optional[str] = None,
        keywords: Optional[Sequence[str]] = None,
        keyword_paths: Optional[Sequence[str | Path]] = None,
        sensitivities: Optional[Sequence[float]] = None,
        device_index: int = -1,
        event_weight: float = 0.9,
    ) -> None:
        load_dotenv()
        resolved_access_key = access_key or os.getenv("PICOVOICE_ACCESS_KEY")
        if not resolved_access_key:
            raise RuntimeError(
                "Hotword detector requires a Picovoice AccessKey. Set PICOVOICE_ACCESS_KEY in the environment "
                "or provide --hotword-access-key."
            )

        if pvporcupine is None or PvRecorder is None:
            raise RuntimeError(
                "Porcupine hotword detection requires `pvporcupine` and `pvrecorder`. Install them with "
                "`pip install pvporcupine pvrecorder`."
            )

        keyword_names: List[str] = []
        porcupine_args: Dict[str, object] = {"access_key": resolved_access_key}

        if keyword_paths:
            normalised_paths = [str(Path(path)) for path in keyword_paths]
            porcupine_args["keyword_paths"] = normalised_paths
            keyword_names.extend(Path(path).stem for path in normalised_paths)

        if keywords:
            keyword_list = list(keywords)
            porcupine_args["keywords"] = keyword_list
            keyword_names.extend(keyword_list)

        if not keyword_names:
            keyword_names = ["picovoice"]
            porcupine_args["keywords"] = keyword_names

        keyword_count = len(keyword_names)
        if sensitivities is not None:
            if len(sensitivities) != keyword_count:
                raise ValueError(
                    "Number of hotword sensitivities must match number of keywords/keyword files."
                )
            porcupine_args["sensitivities"] = list(sensitivities)
        else:
            porcupine_args["sensitivities"] = [0.6] * keyword_count

        try:
            self._porcupine = pvporcupine.create(**porcupine_args)
        except pvporcupine.PorcupineInvalidArgumentError as exc:  # pragma: no cover - library specific error
            raise RuntimeError(f"Invalid Porcupine configuration: {exc}") from exc

        self._keyword_names = keyword_names
        self._event_weight = event_weight
        self._recorder = PvRecorder(device_index=device_index, frame_length=self._porcupine.frame_length)
        self._queue: "Queue[AudioEvent]" = Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._run_exception: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="HotwordDetector", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        try:
            self._recorder.stop()
        except Exception:
            pass
        self._recorder.delete()
        self._porcupine.delete()

    # ------------------------------------------------------------------
    # Event retrieval
    # ------------------------------------------------------------------
    def drain_events(self, *, max_items: int = 8) -> List[AudioEvent]:
        if self._run_exception is not None:
            raise RuntimeError("Hotword detector halted unexpectedly.") from self._run_exception

        events: List[AudioEvent] = []
        for _ in range(max_items):
            try:
                events.append(self._queue.get_nowait())
            except Empty:
                break
        return events

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._recorder.start()
            while not self._stop_event.is_set():
                pcm = self._recorder.read()
                if pcm is None:
                    continue
                keyword_index = self._porcupine.process(pcm)
                if keyword_index >= 0:
                    keyword = self._keyword_names[keyword_index]
                    event = AudioEvent(
                        summary=f"hotword_detected:{keyword}",
                        weight=self._event_weight,
                        data={
                            "type": "audio_hotword",
                            "keyword": keyword,
                            "timestamp": time.time(),
                        },
                    )
                    try:
                        self._queue.put_nowait(event)
                    except Exception:
                        # Queue is full; drop oldest to make room for the newest trigger.
                        try:
                            self._queue.get_nowait()
                        except Empty:
                            pass
                        self._queue.put_nowait(event)
        except BaseException as exc:  # pragma: no cover - background thread safety
            self._run_exception = exc
        finally:
            try:
                self._recorder.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @property
    def keywords(self) -> Sequence[str]:
        return tuple(self._keyword_names)

    def __enter__(self) -> "HotwordDetector":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
