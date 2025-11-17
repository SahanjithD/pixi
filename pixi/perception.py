from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions.hands import HandLandmark


@dataclass(slots=True)
class VisionEvent:
    """Structured description of a perception event."""

    summary: str
    weight: float
    data: Dict[str, float | str | Dict[str, float]]


class VisionProcessor:
    """Captures frames from a camera and extracts face / gesture events."""

    def __init__(
        self,
        *,
        camera_index: int = 0,
        min_face_confidence: float = 0.5,
        min_hand_confidence: float = 0.5,
    ) -> None:
        self._camera_index = camera_index
        self._face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_face_confidence
        )
        self._hands = mp.solutions.hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=min_hand_confidence,
            min_tracking_confidence=min_hand_confidence,
        )

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> List[VisionEvent]:
        """Runs MediaPipe detectors on a BGR frame and returns structured events."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self._face_detector.process(rgb)
        hand_results = self._hands.process(rgb)

        events: List[VisionEvent] = []
        events.extend(self._extract_faces(face_results, frame.shape))
        events.extend(self._extract_hands(hand_results, frame.shape))
        return events

    def _extract_faces(self, results: object, shape: tuple[int, int, int]) -> List[VisionEvent]:
        events: List[VisionEvent] = []
        if not results.detections:
            return events

        height, width, _ = shape
        for detection in results.detections:
            location = detection.location_data.relative_bounding_box
            center_x = location.xmin + (location.width / 2.0)
            center_y = location.ymin + (location.height / 2.0)
            face_area = location.width * location.height
            events.append(
                VisionEvent(
                    summary="face_detected",
                    weight=float(min(1.0, detection.score[0] if detection.score else 0.6)),
                    data={
                        "type": "face",
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "area": float(face_area),
                        "width_px": float(location.width * width),
                        "height_px": float(location.height * height),
                    },
                )
            )
        return events

    def _extract_hands(self, results: object, shape: tuple[int, int, int]) -> List[VisionEvent]:
        events: List[VisionEvent] = []
        if not results.multi_hand_landmarks:
            return events

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness or []
        ):
            gesture = self._classify_gesture(hand_landmarks)
            label = (handedness.classification[0].label if handedness.classification else "unknown").lower()
            events.append(
                VisionEvent(
                    summary="gesture_detected",
                    weight=0.7 if gesture != "unknown" else 0.4,
                    data={"type": "gesture", "gesture": gesture, "hand": label},
                )
            )
        return events

    # ------------------------------------------------------------------
    # Gestures
    # ------------------------------------------------------------------
    def _classify_gesture(self, landmarks: NormalizedLandmarkList) -> str:
        points = landmarks.landmark
        finger_states = self._finger_states(points)

        if finger_states["thumb"] and not any(v for k, v in finger_states.items() if k != "thumb"):
            return "thumbs_up"

        extended_fingers = sum(1 for name, active in finger_states.items() if name != "thumb" and active)
        if extended_fingers >= 4:
            return "open_hand"
        if extended_fingers == 1 and finger_states["index"]:
            return "point"
        if extended_fingers == 0 and not finger_states["thumb"]:
            return "fist"
        return "unknown"

    @staticmethod
    def _finger_states(points: Sequence) -> Dict[str, bool]:
        landmarks = list(points)

        def _is_extended(tip: HandLandmark, pip: HandLandmark) -> bool:
            return landmarks[tip].y < landmarks[pip].y

        thumb_tip = landmarks[HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[HandLandmark.THUMB_IP]
        thumb_cmc = landmarks[HandLandmark.THUMB_CMC]
        thumb_extended = abs(thumb_tip.x - thumb_cmc.x) > 0.05 or abs(thumb_tip.y - thumb_ip.y) > 0.02

        return {
            "thumb": thumb_extended,
            "index": _is_extended(HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_PIP),
            "middle": _is_extended(HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP),
            "ring": _is_extended(HandLandmark.RING_FINGER_TIP, HandLandmark.RING_FINGER_PIP),
            "pinky": _is_extended(HandLandmark.PINKY_TIP, HandLandmark.PINKY_PIP),
        }

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------
    def stream_events(self, *, display: bool = False) -> Generator[List[VisionEvent], None, None]:
        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self._camera_index}")

        window_name = "Pixi Vision" if display else None
        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                events = self.process_frame(frame)

                if display:
                    annotated = frame.copy()
                    self._draw_annotations(annotated, events)
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield events
        finally:
            capture.release()
            if display:
                cv2.destroyAllWindows()

    @staticmethod
    def _draw_annotations(frame: np.ndarray, events: Iterable[VisionEvent]) -> None:
        height, width, _ = frame.shape
        for event in events:
            if event.summary == "face_detected":
                w = int(event.data.get("width_px", 0))
                h = int(event.data.get("height_px", 0))
                cx = int(event.data.get("center_x", 0) * width)
                cy = int(event.data.get("center_y", 0) * height)
                top_left = (int(cx - w / 2), int(cy - h / 2))
                bottom_right = (int(cx + w / 2), int(cy + h / 2))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, "face", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif event.summary == "gesture_detected":
                label = f"{event.data.get('hand', 'hand')}: {event.data.get('gesture', '')}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
