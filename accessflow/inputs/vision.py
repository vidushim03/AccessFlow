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

# Fraction of the screen travelled per unit of normalized nose offset.
# A head turn of roughly 20 degrees should reach a screen edge.
HEAD_X_GAIN = 1.8
HEAD_Y_GAIN = 2.4


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

    def __init__(self, width, height, gaze_enabled, gestures_enabled, gaze_transform=None, head_enabled=False):
        super().__init__()
        self.width, self.height = width, height
        self.gaze_enabled, self.gestures_enabled, self.head_enabled = gaze_enabled, gestures_enabled, head_enabled
        self.stop_requested = False
        self.previous_pinch = False
        self.last_pinch_click = 0.0
        self.pinch_started_at = None
        self.smoothed = None
        self.last_hand_seen = 0.0
        self.last_pointer_time = None
        self.head_origin = None
        self.recenter_requested = False
        self.pointer_filters = {
            "gestures": OneEuroFilter2D(0.4, 0.04),
            "gaze": OneEuroFilter2D(0.12, 0.005),
            "head": OneEuroFilter2D(0.4, 0.04),
        }
        self.gaze_transform = np.asarray(gaze_transform, dtype=float) if gaze_transform is not None else None

    def _smooth_pointer(self, source, x, y, max_step_px, deadzone_px):
        now = time.monotonic()
        filtered = self.pointer_filters[source].update((x, y), now)
        # Keep pointer speed roughly constant when the camera runs below 30 FPS.
        frame_scale = 1.0
        if self.last_pointer_time is not None:
            frame_scale = max(0.5, min(30.0, (now - self.last_pointer_time) * 30.0))
        self.last_pointer_time = now
        current = self.smoothed
        if current is None:
            current = filtered
        else:
            dx = (filtered[0] - current[0]) * self.width
            dy = (filtered[1] - current[1]) * self.height
            if math.hypot(dx, dy) < deadzone_px:
                return current
            step = max_step_px * frame_scale
            dx = max(-step, min(step, dx))
            dy = max(-step, min(step, dy))
            current = (max(0.0, min(1.0, current[0] + dx / self.width)),
                       max(0.0, min(1.0, current[1] + dy / self.height)))
        self.smoothed = current
        return current

    @staticmethod
    def _extract_head_feature(landmarks):
        """Nose-tip offset from the face centre, normalized by face size."""
        nose = landmarks[1]
        left, right = landmarks[234], landmarks[454]
        top, chin = landmarks[10], landmarks[152]
        face_width = math.hypot(right.x - left.x, right.y - left.y)
        face_height = math.hypot(chin.x - top.x, chin.y - top.y)
        if face_width < 1e-6 or face_height < 1e-6:
            return None
        centre_x = (left.x + right.x) / 2.0
        centre_y = (left.y + right.y) / 2.0
        return (nose.x - centre_x) / face_width, (nose.y - centre_y) / face_height

    def _head_pointer(self, feature):
        """Map head offset to screen coordinates relative to the first neutral pose."""
        if self.head_origin is None:
            self.head_origin = feature
            return (0.5, 0.5)
        dx = (feature[0] - self.head_origin[0]) * HEAD_X_GAIN
        dy = (feature[1] - self.head_origin[1]) * HEAD_Y_GAIN
        return (max(0.0, min(1.0, 0.5 + dx)), max(0.0, min(1.0, 0.5 + dy)))

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

    def recenter_head(self):
        self.recenter_requested = True

    def stop(self):
        """Ask the worker to stop without blocking the caller.

        MediaPipe teardown (face/hand close) can take seconds, so callers on the
        GUI thread must not wait for the thread here; they keep the worker alive
        until its ``finished`` signal arrives.
        """
        self.stop_requested = True

    def run(self):
        capture = face = hand = None
        reported_stop = False
        try:
            capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not capture.isOpened():
                self.status.emit("Camera unavailable")
                reported_stop = True
                return
            if self.gaze_enabled or self.head_enabled:
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
            last_face_seen = time.monotonic()
            last_face_warning = 0.0
            face_seen = False
            last_brightness_check = 0.0
            dark_checks = 0
            dark_warned = False
            failed_reads = 0
            while not self.stop_requested:
                ok, frame = capture.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= 30:
                        self.status.emit("Camera stopped delivering frames")
                        reported_stop = True
                        break
                    time.sleep(0.02)
                    continue
                failed_reads = 0
                now = time.monotonic()
                if now - last_brightness_check >= 1.0:
                    last_brightness_check = now
                    if float(frame.mean()) < 8.0:
                        dark_checks += 1
                        if dark_checks == 2 and not dark_warned:
                            self.status.emit("Camera image is dark - check the privacy shutter and lighting")
                            dark_warned = True
                    else:
                        if dark_warned:
                            self.status.emit("Camera image restored")
                            dark_warned = False
                        dark_checks = 0
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
                        last_face_seen = now
                        if not face_seen or last_face_warning:
                            self.status.emit("Face tracking active")
                            face_seen = True
                            last_face_warning = 0.0
                        landmarks = faces[0]
                        gaze_active = False
                        if self.gaze_enabled:
                            gaze_feat = self._extract_gaze_feature(landmarks)
                            if gaze_feat is not None:
                                rx, ry = gaze_feat
                                self.gaze_sample.emit((rx, ry))
                                if self.gaze_transform is not None:
                                    raw = np.array([rx, ry, 1.0]) @ self.gaze_transform
                                    raw = (max(0.0, min(1.0, float(raw[0]))), max(0.0, min(1.0, float(raw[1]))))
                                    sx, sy = self._smooth_pointer("gaze", raw[0], raw[1], 25.0, 4.0)
                                    gaze_active = True
                                    self.event_ready.emit(InteractionEvent("gaze", "gaze", "move_pointer", 0.82, {
                                        "x": int(sx * (self.width - 1)), "y": int(sy * (self.height - 1))}))
                        if self.head_enabled and not gaze_active:
                            if self.recenter_requested:
                                self.head_origin = None
                                self.pointer_filters["head"].reset()
                                self.recenter_requested = False
                            head_feat = self._extract_head_feature(landmarks)
                            if head_feat is not None:
                                point = self._head_pointer(head_feat)
                                if point is not None:
                                    sx, sy = self._smooth_pointer("head", point[0], point[1], 25.0, 4.0)
                                    self.event_ready.emit(InteractionEvent("head", "head", "move_pointer", 0.75, {
                                        "x": int(sx * (self.width - 1)), "y": int(sy * (self.height - 1))}))
                    elif now - last_face_seen > 3.0 and now - last_face_warning > 10.0:
                        self.status.emit("No face detected - check camera framing and lighting")
                        last_face_warning = now
                elif hand_seen:
                    # Clear face-cursor filter state when a hand is active so there is no residual drift on release
                    self.pointer_filters["gaze"].reset()
                    self.pointer_filters["head"].reset()
        except Exception as exc:
            self.status.emit(f"Camera stopped: {exc}")
            reported_stop = True
        finally:
            if face:
                face.close()
            if hand:
                hand.close()
            if capture:
                capture.release()
            if not reported_stop:
                self.status.emit("Camera inputs stopped")

