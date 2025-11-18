from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions.hands import HandLandmark

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.face_detector import FaceDetectorOptions

    _TASKS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    mp_python = None
    mp_vision = None
    BaseOptions = None
    FaceDetectorOptions = None
    _TASKS_AVAILABLE = False


def _has_tasks_image() -> bool:
    vision_module = getattr(mp_python, "vision", None)
    return vision_module is not None and hasattr(vision_module, "Image")


@dataclass(slots=True)
class VisionEvent:
    """Structured description of a perception event."""

    summary: str
    weight: float
    data: Dict[str, float | str | bool | Dict[str, float]]


@dataclass(slots=True)
class TrackedFace:
    """Internal helper for lightweight face tracking."""

    face_id: int
    center_x: float
    center_y: float
    area: float
    confidence: float
    last_seen: float


class VisionProcessor:
    """Captures frames from a camera and extracts face / gesture events."""

    def __init__(
        self,
        *,
        camera_index: int = 0,
        min_face_confidence: float = 0.5,
        min_hand_confidence: float = 0.5,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        enable_hands: bool = True,
        hand_detection_interval: int = 5,
        enable_face_tracking: bool = True,
        smoothing_factor: float = 0.6,
        tracking_timeout: float = 1.2,
        face_detection_interval: int = 10,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
        frame_callback_use_annotations: bool = True,
    ) -> None:
        self._min_face_confidence = min_face_confidence
        self._camera_index = camera_index
        self._frame_width = frame_width or 320
        self._frame_height = frame_height or 240
        self._enable_hands = enable_hands
        self._hand_interval = max(1, hand_detection_interval)
        self._frame_counter = 0
        self._enable_face_tracking = enable_face_tracking
        self._smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self._tracking_timeout = tracking_timeout
        self._face_detection_interval = max(1, face_detection_interval)
        self._frames_since_face_detection = self._face_detection_interval
        self._face_id_counter = 0
        self._tracked_face: Optional[TrackedFace] = None
        self._last_primary_face_box: Optional[Tuple[float, float]] = None
        self._frame_callback = frame_callback
        self._frame_callback_use_annotations = frame_callback_use_annotations

        self._use_tasks_detector = False
        model_path = Path(__file__).with_name("blaze_face_short_range.tflite")
        if not model_path.exists():
            model_path = Path(__file__).resolve().parent.parent / "blaze_face_short_range.tflite"
        if _TASKS_AVAILABLE and _has_tasks_image() and model_path.exists():
            try:
                base_options = BaseOptions(model_asset_path=str(model_path))
                options = FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=min_face_confidence,
                    min_suppression_threshold=min_face_confidence,
                )
                self._face_detector = mp_vision.FaceDetector.create_from_options(options)
                self._use_tasks_detector = True
            except Exception:
                self._face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self._min_face_confidence,
                )
        else:
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self._min_face_confidence,
            )

        if enable_hands:
            self._hands = mp.solutions.hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=min_hand_confidence,
                min_tracking_confidence=min_hand_confidence,
            )
        else:
            self._hands = None

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, *, detect_hands: bool = True) -> List[VisionEvent]:
        """Runs MediaPipe detectors on a BGR frame and returns structured events."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections: Optional[List] = None
        detection_performed = False
        if (
            not self._enable_face_tracking
            or self._tracked_face is None
            or self._frames_since_face_detection >= self._face_detection_interval
        ):
            detections = self._run_face_detector(rgb)
            detection_performed = True
            self._frames_since_face_detection = 0
        else:
            self._frames_since_face_detection += 1

        hand_results = None
        if detect_hands and self._hands is not None:
            hand_results = self._hands.process(rgb)

        events: List[VisionEvent] = []
        face_events = self._extract_faces(detections, frame.shape, detection_performed)
        if not face_events:
            fallback_event = self._build_tracked_face_event(frame.shape)
            if fallback_event:
                face_events.append(fallback_event)
        else:
            primary_data = face_events[0].data
            self._last_primary_face_box = (
                float(primary_data.get("width_px", 0.0)),
                float(primary_data.get("height_px", 0.0)),
            )
        events.extend(face_events)
        events.extend(self._extract_hands(hand_results, frame.shape))
        return events

    def _run_face_detector(self, rgb: np.ndarray) -> List:
        if self._use_tasks_detector:
            try:
                mp_image = mp_python.vision.Image(
                    image_format=mp_python.vision.ImageFormat.SRGB,
                    data=rgb,
                )
            except AttributeError:
                # Older MediaPipe build lacks the Image class; fall back to classical API.
                self._use_tasks_detector = False
                self._face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self._min_face_confidence,
                )
            else:
                result = self._face_detector.detect(mp_image)
                return list(getattr(result, "detections", []) or [])
        result = self._face_detector.process(rgb)
        return list(getattr(result, "detections", []) or [])

    def _extract_faces(
        self,
        detections: Optional[Iterable],
        shape: tuple[int, int, int],
        detection_performed: bool,
    ) -> List[VisionEvent]:
        events: List[VisionEvent] = []

        if not detections:
            if (
                detection_performed
                and self._enable_face_tracking
                and self._tracked_face
                and time.time() - self._tracked_face.last_seen > self._tracking_timeout
            ):
                self._tracked_face = None
            return events

        height, width, _ = shape

        def _score(det: object) -> float:
            if self._use_tasks_detector:
                return det.categories[0].score if det.categories else 0.0
            return det.score[0] if det.score else 0.0

        sorted_detections = sorted(detections, key=_score, reverse=True)

        for index, detection in enumerate(sorted_detections):
            if self._use_tasks_detector:
                bbox = detection.bounding_box
                width_px = float(bbox.width)
                height_px = float(bbox.height)
                center_x = (bbox.origin_x + width_px / 2.0) / width
                center_y = (bbox.origin_y + height_px / 2.0) / height
                face_area = max(0.0, (width_px / width) * (height_px / height))
                confidence = float(detection.categories[0].score if detection.categories else 0.6)
            else:
                location = detection.location_data.relative_bounding_box
                center_x = location.xmin + (location.width / 2.0)
                center_y = location.ymin + (location.height / 2.0)
                face_area = max(0.0, location.width * location.height)
                confidence = float(min(1.0, detection.score[0] if detection.score else 0.6))
                width_px = float(location.width * width)
                height_px = float(location.height * height)

            tracking_info: Optional[TrackedFace] = None
            if self._enable_face_tracking:
                if index == 0:
                    tracking_info = self._update_tracked_face(center_x, center_y, face_area, confidence)
                elif self._tracked_face:
                    tracking_info = self._tracked_face

            event_data: Dict[str, float | str | bool | Dict[str, float]] = {
                "type": "face",
                "center_x": float(center_x),
                "center_y": float(center_y),
                "area": float(face_area),
                "width_px": width_px,
                "height_px": height_px,
                "confidence": confidence,
                "is_primary": index == 0,
            }

            if tracking_info:
                event_data.update(
                    {
                        "tracking_id": tracking_info.face_id,
                        "smoothed_center_x": tracking_info.center_x,
                        "smoothed_center_y": tracking_info.center_y,
                        "smoothed_area": tracking_info.area,
                        "tracking_confidence": tracking_info.confidence,
                    }
                )

            events.append(
                VisionEvent(
                    summary="face_detected",
                    weight=confidence,
                    data=event_data,
                )
            )

        return events

    def _build_tracked_face_event(self, shape: tuple[int, int, int]) -> Optional[VisionEvent]:
        if not self._enable_face_tracking or not self._tracked_face:
            return None
        now = time.time()
        if now - self._tracked_face.last_seen > self._tracking_timeout:
            self._tracked_face = None
            return None

        height, width, _ = shape
        if self._last_primary_face_box:
            width_px, height_px = self._last_primary_face_box
        else:
            width_px = float(self._tracked_face.area * width)
            height_px = float(self._tracked_face.area * height)

        data: Dict[str, float | str | bool] = {
            "type": "face",
            "center_x": float(self._tracked_face.center_x),
            "center_y": float(self._tracked_face.center_y),
            "area": float(self._tracked_face.area),
            "width_px": float(width_px),
            "height_px": float(height_px),
            "confidence": float(self._tracked_face.confidence),
            "is_primary": True,
            "tracking_id": self._tracked_face.face_id,
            "smoothed_center_x": float(self._tracked_face.center_x),
            "smoothed_center_y": float(self._tracked_face.center_y),
            "smoothed_area": float(self._tracked_face.area),
            "tracking_confidence": float(self._tracked_face.confidence),
        }
        self._tracked_face.last_seen = now
        return VisionEvent(summary="face_detected", weight=self._tracked_face.confidence, data=data)

    def _update_tracked_face(self, center_x: float, center_y: float, area: float, confidence: float) -> TrackedFace:
        now = time.time()
        if self._tracked_face is None:
            self._face_id_counter += 1
            self._tracked_face = TrackedFace(
                face_id=self._face_id_counter,
                center_x=center_x,
                center_y=center_y,
                area=area,
                confidence=confidence,
                last_seen=now,
            )
            return self._tracked_face

        distance = float(np.hypot(center_x - self._tracked_face.center_x, center_y - self._tracked_face.center_y))
        if distance > 0.25:
            self._face_id_counter += 1
            self._tracked_face = TrackedFace(
                face_id=self._face_id_counter,
                center_x=center_x,
                center_y=center_y,
                area=area,
                confidence=confidence,
                last_seen=now,
            )
            return self._tracked_face

        alpha = self._smoothing_factor
        smoothed_center_x = (alpha * center_x) + ((1 - alpha) * self._tracked_face.center_x)
        smoothed_center_y = (alpha * center_y) + ((1 - alpha) * self._tracked_face.center_y)
        smoothed_area = (0.2 * area) + (0.8 * self._tracked_face.area)

        self._tracked_face.center_x = smoothed_center_x
        self._tracked_face.center_y = smoothed_center_y
        self._tracked_face.area = smoothed_area
        self._tracked_face.confidence = confidence
        self._tracked_face.last_seen = now
        return self._tracked_face

    def _extract_hands(self, results: object, shape: tuple[int, int, int]) -> List[VisionEvent]:
        events: List[VisionEvent] = []
        if not results or not getattr(results, "multi_hand_landmarks", None):
            return events

        handedness_list = list(results.multi_handedness or [])
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            gesture = self._classify_gesture(hand_landmarks)
            label = "unknown"
            if index < len(handedness_list):
                classification = handedness_list[index].classification
                if classification:
                    label = classification[0].label.lower()
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

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._frame_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._frame_height))

        window_name = "Pixi Vision" if display else None
        if not display and self._frame_callback is not None:
            window_name = None
        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                self._frame_counter += 1
                processing_frame = cv2.resize(
                    frame,
                    (self._frame_width, self._frame_height),
                    interpolation=cv2.INTER_AREA,
                )

                detect_hands = self._enable_hands and (self._frame_counter % self._hand_interval == 0)
                events = self.process_frame(processing_frame, detect_hands=detect_hands)

                annotated_frame: Optional[np.ndarray] = None
                if display or (self._frame_callback is not None and self._frame_callback_use_annotations):
                    annotated_frame = processing_frame.copy()
                    self._draw_annotations(annotated_frame, events)

                if display and annotated_frame is not None:
                    cv2.imshow(window_name, annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if self._frame_callback is not None:
                    frame_for_callback = annotated_frame if annotated_frame is not None else processing_frame
                    self._frame_callback(frame_for_callback.copy())

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
                cx_norm = float(event.data.get("smoothed_center_x", event.data.get("center_x", 0.5)))
                cy_norm = float(event.data.get("smoothed_center_y", event.data.get("center_y", 0.5)))
                cx = int(cx_norm * width)
                cy = int(cy_norm * height)
                top_left = (int(cx - w / 2), int(cy - h / 2))
                bottom_right = (int(cx + w / 2), int(cy + h / 2))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                label = "face"
                if event.data.get("tracking_id") is not None:
                    label = f"face #{event.data['tracking_id']}"
                cv2.putText(frame, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                coord_label = (
                    f"x={cx_norm:.2f}, y={cy_norm:.2f}, area="
                    f"{float(event.data.get('smoothed_area', event.data.get('area', 0.0))):.3f}"
                )
                cv2.putText(
                    frame,
                    coord_label,
                    (top_left[0], bottom_right[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            elif event.summary == "gesture_detected":
                label = f"{event.data.get('hand', 'hand')}: {event.data.get('gesture', '')}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
