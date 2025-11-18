from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

_HTML_TEMPLATE = """<!doctype html>\n<html lang=\"en\">\n  <head>\n    <meta charset=\"utf-8\">\n    <title>{title}</title>\n    <style>\n      body {{ background: #111; color: #ddd; font-family: sans-serif; text-align: center; }}\n      img {{ max-width: 90vw; height: auto; border: 2px solid #444; margin-top: 2rem; }}\n      h1 {{ font-size: 1.4rem; margin-top: 1.5rem; }}\n    </style>\n  </head>\n  <body>\n    <h1>{title}</h1>\n    <img src=\"/stream\" alt=\"Live camera preview\" />\n  </body>\n</html>\n"""


@dataclass
class FrameBroadcaster:
    """Thread-safe container for the latest annotated frame."""

    _latest_frame: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame.copy()

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()


def create_app(broadcaster: FrameBroadcaster, *, title: str = "Pixi Camera Preview") -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return render_template_string(_HTML_TEMPLATE.format(title=title))

    @app.route("/stream")
    def stream() -> Response:
        def _generator():
            while True:
                frame = broadcaster.latest()
                if frame is None:
                    time.sleep(0.05)
                    continue

                ok, buffer = cv2.imencode(".jpg", frame)
                if not ok:
                    continue

                payload = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")

        return Response(
            _generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def start_preview_server(
    broadcaster: FrameBroadcaster,
    *,
    host: str = "0.0.0.0",
    port: int = 5000,
    title: str = "Pixi Camera Preview",
) -> threading.Thread:
    app = create_app(broadcaster, title=title)

    def _run() -> None:
        app.run(host=host, port=port, debug=False, threaded=True)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
