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
        self.smoothed = None
        self.gaze_transform = np.asarray(gaze_transform, dtype=float) if gaze_transform is not None else None

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
                    running_mode=RunningMode.VIDEO, num_faces=1))
            if self.gestures_enabled:
                hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_model_path("hand_landmarker.task", HAND_URL)),
                    running_mode=RunningMode.VIDEO, num_hands=1))
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
                        points = hands[0]
                        index_tip, thumb_tip = points[8], points[4]
                        x = int(max(0, min(1, index_tip.x)) * (self.width - 1))
                        y = int(max(0, min(1, index_tip.y)) * (self.height - 1))
                        self.event_ready.emit(InteractionEvent("gestures", "pointer", "move_pointer", 0.86, {"x": x, "y": y}))
                        pinch = ((index_tip.x-thumb_tip.x)**2 + (index_tip.y-thumb_tip.y)**2)**0.5 < 0.055
                        if pinch and not self.previous_pinch:
                            self.event_ready.emit(InteractionEvent("gestures", "pinch", "click", 0.90, {"button": "left"}))
                        self.previous_pinch = pinch
                # A visible hand owns the camera interaction for this frame; this avoids gaze/gesture cursor contention.
                if face and not hand_seen:
                    faces = face.detect_for_video(image, timestamp).face_landmarks
                    if faces:
                        marks = faces[0]
                        ratios = []
                        for iris, aidx, bidx, upper, lower in ((468,33,133,159,145),(473,362,263,386,374)):
                            a, b, eye = marks[aidx], marks[bidx], marks[iris]
                            width = abs(b.x-a.x)
                            if width > 0.002:
                                ratios.append(((eye.x-min(a.x,b.x))/width,
                                               (eye.y-(marks[upper].y+marks[lower].y)/2)/width))
                        if ratios:
                            rx = sum(v[0] for v in ratios)/len(ratios)
                            ry = sum(v[1] for v in ratios)/len(ratios)
                            self.gaze_sample.emit((rx, ry))
                            if self.gaze_transform is not None:
                                raw = np.array([rx, ry, 1.0]) @ self.gaze_transform
                                raw = (max(0.0, min(1.0, float(raw[0]))), max(0.0, min(1.0, float(raw[1]))))
                                self.smoothed = raw if self.smoothed is None else tuple(.22*raw[i]+.78*self.smoothed[i] for i in (0,1))
                                self.event_ready.emit(InteractionEvent("gaze", "gaze", "move_pointer", 0.82, {
                                    "x": int(self.smoothed[0]*(self.width-1)), "y": int(self.smoothed[1]*(self.height-1))}))
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




