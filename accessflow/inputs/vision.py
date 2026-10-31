import math
import os
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QThread, Signal
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, HandLandmarker, HandLandmarkerOptions, RunningMode
from accessflow.core.events import InteractionEvent
from accessflow.inputs.filters import OneEuroFilter2D

FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def _model_path(name, url):
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AccessFlow" / "models"
    target = base / name
    if not target.exists():
        base.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".download")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    return str(target)


class VisionWorker(QThread):
    event_ready = Signal(object)
    gaze_sample = Signal(object)
    status = Signal(str)

    def __init__(self, width, height, gaze_enabled, gestures_enabled, gaze_transform=None):
        super().__init__()
        self.width, self.height = width, height
        self.gaze_enabled, self.gestures_enabled = gaze_enabled, gestures_enabled
        self.stop_requested = False
        self.previous_pinch = False
        self.last_pinch_click = 0.0
        self.pinch_started_at = None
        self.smoothed = None
        self.last_hand_seen = 0.0
        self.pointer_filters = {
            "gestures": OneEuroFilter2D(0.4, 0.06),
            "gaze": OneEuroFilter2D(0.35, 0.035),
        }
        self.gaze_transform = np.asarray(gaze_transform, dtype=float) if gaze_transform is not None else None

    def _smooth_pointer(self, source, x, y, max_step_px, deadzone_px):
        filtered = self.pointer_filters[source].update((x, y), time.monotonic())
        current = self.smoothed
        if current is None:
            current = filtered
        else:
            dx = (filtered[0] - current[0]) * self.width
            dy = (filtered[1] - current[1]) * self.height
            if math.hypot(dx, dy) < deadzone_px:
                return current
            dx = max(-max_step_px, min(max_step_px, dx))
            dy = max(-max_step_px, min(max_step_px, dy))
            current = (max(0.0, min(1.0, current[0] + dx / self.width)),
                       max(0.0, min(1.0, current[1] + dy / self.height)))
        self.smoothed = current
        return current

    @staticmethod
    def _extract_gaze_feature(marks):
        """Invariant 2D gaze feature in eye-corner coordinates.
        Computes iris position along the eye-corner axis and perpendicular offset,
        making tracking robust against head tilt, distance changes, and roll."""
        features = []
        # Left eye: iris=468, corners outer=33, inner=133
        # Right eye: iris=473, corners inner=362, outer=263
        for iris_idx, a_idx, b_idx in ((468, 33, 133), (473, 362, 263)):
            iris, a, b = marks[iris_idx], marks[a_idx], marks[b_idx]
            axis_x = b.x - a.x
            axis_y = b.y - a.y
            width_sq = axis_x * axis_x + axis_y * axis_y
            if width_sq < 1e-6:
                continue
            rel_x = iris.x - a.x
            rel_y = iris.y - a.y
            x = (rel_x * axis_x + rel_y * axis_y) / width_sq
            y = (axis_x * rel_y - axis_y * rel_x) / width_sq
            features.append((x, y))
        if not features:
            return None
        return (sum(f[0] for f in features) / len(features),
                sum(f[1] for f in features) / len(features))

    def stop(self):
        self.stop_requested = True
        self.wait(3000)

    def run(self):
        capture = face = hand = None
        try:
            capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not capture.isOpened():
                self.status.emit("Camera unavailable")
                return
            if self.gaze_enabled:
                face = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_model_path("face_landmarker.task", FACE_URL)),
                    running_mode=RunningMode.VIDEO, num_faces=1,
                    min_face_detection_confidence=0.70, min_face_presence_confidence=0.60,
                    min_tracking_confidence=0.65))
            if self.gestures_enabled:
                hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_model_path("hand_landmarker.task", HAND_URL)),
                    running_mode=RunningMode.VIDEO, num_hands=1, min_hand_detection_confidence=0.70,
                    min_hand_presence_confidence=0.60, min_tracking_confidence=0.65))
            self.status.emit("Camera inputs active")
            timestamp = int(time.monotonic() * 1000)
            while not self.stop_requested:
                ok, frame = capture.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp += 34
                hand_seen = False
                if hand:
                    hands = hand.detect_for_video(image, timestamp).hand_landmarks
                    if hands:
                        hand_seen = True
                        self.last_hand_seen = time.monotonic()
                        points = hands[0]
                        index_tip, thumb_tip = points[8], points[4]
                        sx, sy = self._smooth_pointer("gestures", index_tip.x, index_tip.y, 70.0, 3.0)
                        x = int(sx * (self.width - 1))
                        y = int(sy * (self.height - 1))
                        self.event_ready.emit(InteractionEvent("gestures", "pointer", "move_pointer", 0.86, {"x": x, "y": y}))
                        palm_size = math.hypot(points[0].x - points[9].x, points[0].y - points[9].y)
                        pinch_ratio = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y) / max(palm_size, 0.001)
                        now = time.monotonic()
                        if not self.previous_pinch and pinch_ratio <= 0.32:
                            if self.pinch_started_at is None:
                                self.pinch_started_at = now
                            elif now - self.pinch_started_at >= 0.08 and now - self.last_pinch_click >= 0.70:
                                self.event_ready.emit(InteractionEvent("gestures", "pinch", "click", 0.90, {"button": "left"}))
                                self.previous_pinch = True
                                self.last_pinch_click = now
                        elif self.previous_pinch and pinch_ratio >= 0.48:
                            self.previous_pinch = False
                            self.pinch_started_at = None
                        elif not self.previous_pinch and pinch_ratio > 0.32:
                            self.pinch_started_at = None
                # A visible hand owns the camera interaction for this frame; this avoids gaze/gesture cursor contention.
                if face and not hand_seen and (not self.gestures_enabled or time.monotonic() - self.last_hand_seen > 0.5):
                    faces = face.detect_for_video(image, timestamp).face_landmarks
                    if faces:
                        gaze_feat = self._extract_gaze_feature(faces[0])
                        if gaze_feat is not None:
                            rx, ry = gaze_feat
                            self.gaze_sample.emit((rx, ry))
                            if self.gaze_transform is not None:
                                raw = np.array([rx, ry, 1.0]) @ self.gaze_transform
                                raw = (max(0.0, min(1.0, float(raw[0]))), max(0.0, min(1.0, float(raw[1]))))
                                sx, sy = self._smooth_pointer("gaze", raw[0], raw[1], 35.0, 2.0)
                                self.event_ready.emit(InteractionEvent("gaze", "gaze", "move_pointer", 0.82, {
                                    "x": int(sx * (self.width - 1)), "y": int(sy * (self.height - 1))}))
                elif hand_seen:
                    # Clear gaze filter state when hand is active so there is no residual drift on hand release
                    self.pointer_filters["gaze"].reset()
        except Exception as exc:
            self.status.emit(f"Camera stopped: {exc}")
        finally:
            if face:
                face.close()
            if hand:
                hand.close()
            if capture:
                capture.release()
            self.status.emit("Camera inputs stopped")

