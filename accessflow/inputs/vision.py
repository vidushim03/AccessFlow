import ctypes
import math
import os
import sys
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

if sys.platform == "win32":
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
    _user32.GetCursorPos.restype = wintypes.BOOL
else:
    _user32 = None

FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Head tracking mirrors the reference tracker in vision-access-hci: the raw
# nose-tip position is compared with a learned neutral pose and scaled.
# The value is a fraction of the screen, so 0.1 of head travel moves the
# pointer a tenth of the screen. Horizontal gain is lower than vertical
# because lateral head rotation produces smaller landmark displacement.
HEAD_X_GAIN = 1.8
HEAD_Y_GAIN = 2.4
# Average the first few samples so a single noisy first frame cannot become the
# permanent neutral pose.
HEAD_NEUTRAL_SAMPLES = 8
# A single frame that claims the pointer should teleport across more than this
# fraction of the screen is a landmark glitch, not real motion.
POINTER_JUMP_LIMIT = 0.35
# EMA alpha for head tracking, matching vision-access-hci's 0.18 per frame.
# The reference uses a plain EMA on absolute position with no deadzone and no
# per-frame step clamp; that heavy low-pass is what keeps a still head from
# dragging the pointer.
HEAD_EMA_ALPHA = 0.18


class HeadTracker:
    """Head-pointer controller mirroring vision-access-hci.

    The raw nose-tip landmark is compared with a learned neutral pose and
    scaled by the configured gains to produce an absolute target position,
    which is then passed through a plain EMA. There is no offset anchoring,
    no per-frame step clamp and no deadzone: the EMA alone is the low-pass,
    and it settles to the anchor when the head is still, so a still user
    produces no pointer drift.

    The tracker can be re-anchored to an arbitrary pointer position (used
    after a touchpad takeover or input-method switch) by resetting the
    smoothed value and re-deriving the neutral pose from the current feature.
    """

    def __init__(self, alpha=HEAD_EMA_ALPHA, gain_x=HEAD_X_GAIN,
                 gain_y=HEAD_Y_GAIN, neutral_samples=HEAD_NEUTRAL_SAMPLES):
        self.alpha = alpha
        self.gain_x = gain_x
        self.gain_y = gain_y
        self.origin = None
        self._samples = []
        self.smoothed = None

    def reset(self):
        """Forget the learned neutral pose and smoothed position."""
        self.origin = None
        self._samples = []
        self.smoothed = None

    def _learn_origin(self, feature):
        self._samples.append(feature)
        if len(self._samples) < HEAD_NEUTRAL_SAMPLES:
            return False
        count = len(self._samples)
        self.origin = (sum(s[0] for s in self._samples) / count,
                       sum(s[1] for s in self._samples) / count)
        self._samples = []
        return True

    def update(self, feature, anchor=None):
        """Map a nose-tip feature to a smoothed pointer position.

        Args:
            feature: (x, y) raw nose-tip landmark in normalised image space.
            anchor: Optional (x, y) normalised screen position to anchor the
                tracker to when it has no smoothed state yet. Used after a
                touchpad takeover, recenter, or first activation so the pointer
                does not jump.

        Returns:
            (x, y) normalised screen position, or None while learning origin.
        """
        if self.smoothed is None and anchor is not None:
            # Anchor to an external pointer position: derive the neutral pose
            # so the current head position maps to the anchor, then start the
            # EMA there. This is what "Recenter head tracking" and the
            # touchpad takeover do — the pointer stays where it is.
            self.origin = (feature[0] - (anchor[0] - 0.5) / self.gain_x,
                           feature[1] - (anchor[1] - 0.5) / self.gain_y)
            self.smoothed = (max(0.0, min(1.0, anchor[0])),
                             max(0.0, min(1.0, anchor[1])))
            return self.smoothed
        if self.origin is None:
            if not self._learn_origin(feature):
                return None
        dx = (feature[0] - self.origin[0]) * self.gain_x
        dy = (feature[1] - self.origin[1]) * self.gain_y
        target = (max(0.0, min(1.0, 0.5 + dx)),
                  max(0.0, min(1.0, 0.5 + dy)))
        if self.smoothed is None:
            self.smoothed = target
        else:
            self.smoothed = (
                self.alpha * target[0] + (1.0 - self.alpha) * self.smoothed[0],
                self.alpha * target[1] + (1.0 - self.alpha) * self.smoothed[1],
            )
        return self.smoothed


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
        self.last_hand_seen = 0.0
        self.pointer_suspended_until = 0.0
        self.pointer_was_suspended = False
        self.pointer_source = None
        self.pointer_output = {}
        self.pointer_offset = {}
        self.pointer_last = None
        self.head_tracker = HeadTracker()
        self.recenter_requested = False
        self.pointer_filters = {
            # Fast enough to follow the hand, smooth enough to hide landmark jitter.
            "gestures": OneEuroFilter2D(1.2, 0.02),
            # Eye gaze is inherently noisy and calibrated; keep it slow.
            "gaze": OneEuroFilter2D(0.15, 0.005),
            # Comparable to the reference EMA (0.18 at 60 FPS) plus speed adaptation.
            "head": OneEuroFilter2D(0.9, 0.03),
        }
        self.gaze_transform = np.asarray(gaze_transform, dtype=float) if gaze_transform is not None else None

    def _screen_pointer(self):
        """Current real pointer position in normalised screen coordinates."""
        if _user32 is None:
            return None
        try:
            point = wintypes.POINT()
            if not _user32.GetCursorPos(ctypes.byref(point)):
                return None
        except Exception:
            return None
        return (max(0.0, min(1.0, point.x / max(1, self.width - 1))),
                max(0.0, min(1.0, point.y / max(1, self.height - 1))))

    def _forget_source(self, source):
        """Discard smoothed state for one source so it restarts cleanly."""
        self.pointer_filters[source].reset()
        self.pointer_output.pop(source, None)
        self.pointer_offset.pop(source, None)
        if self.pointer_source == source:
            self.pointer_source = None

    def _reset_pointer_state(self):
        """Forget every source, e.g. after the touchpad has moved the pointer."""
        self.pointer_source = None
        self.pointer_output.clear()
        self.pointer_offset.clear()
        # The head tracker holds its own smoothed state; drop it so the next
        # head frame re-anchors to wherever the pointer now is.
        self.head_tracker.smoothed = None

    def _smooth_pointer(self, source, x, y, deadzone_px, absolute=False):
        """Filter one pointer sample and return the screen position to use.

        Head and hand control are relative to where the pointer already is: on
        first use, after an input-method switch, or after a touchpad takeover,
        the pointer is anchored rather than jumped. Gaze is calibrated to
        absolute screen coordinates, so it is never anchored.
        """
        now = time.monotonic()
        switching = self.pointer_source != source
        if switching:
            # Never let a filter carry motion across an input-method change.
            self.pointer_filters[source].reset()
            self.pointer_output.pop(source, None)
        filtered = self.pointer_filters[source].update((x, y), now)
        previous = self.pointer_output.get(source)
        if previous is None:
            if absolute:
                current = (max(0.0, min(1.0, filtered[0])),
                           max(0.0, min(1.0, filtered[1])))
            else:
                current = self._screen_pointer() or self.pointer_last or (0.5, 0.5)
            self.pointer_offset[source] = (current[0] - filtered[0], current[1] - filtered[1])
            self.pointer_output[source] = current
            self.pointer_source = source
            self.pointer_last = current
            return current
        target_x = filtered[0] + self.pointer_offset[source][0]
        target_y = filtered[1] + self.pointer_offset[source][1]
        dx = (target_x - previous[0]) * self.width
        dy = (target_y - previous[1]) * self.height
        distance = math.hypot(dx, dy)
        if distance < deadzone_px:
            return previous
        if distance > self.width * POINTER_JUMP_LIMIT:
            # A bad landmark frame can claim a huge jump. Keep the pointer still
            # and rebase the source on the new sample so normal movement can
            # continue on the next frame.
            self.pointer_offset[source] = (previous[0] - filtered[0],
                                           previous[1] - filtered[1])
            return previous
        current = (max(0.0, min(1.0, previous[0] + dx / self.width)),
                   max(0.0, min(1.0, previous[1] + dy / self.height)))
        self.pointer_output[source] = current
        self.pointer_last = current
        return current

    @staticmethod
    def _extract_head_feature(landmarks):
        """Raw nose-tip position, as used by the reference tracker."""
        nose = landmarks[1]
        if not (math.isfinite(nose.x) and math.isfinite(nose.y)):
            return None
        return (nose.x, nose.y)

    def _head_pointer(self, feature, anchor=None):
        """Map the nose position to a smoothed pointer position.

        Delegates to the HeadTracker, which mirrors the reference
        vision-access-hci algorithm (raw nose vs neutral, EMA 0.18). The
        anchor is the current real pointer position when head tracking is
        first activated or re-anchored after a touchpad takeover, so the
        pointer does not jump.
        """
        return self.head_tracker.update(feature, anchor=anchor)

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

    def suspend_pointer(self, seconds=1.5):
        """Hand the pointer back to the touchpad/mouse for a short while."""
        self.pointer_suspended_until = max(
            self.pointer_suspended_until, time.monotonic() + seconds)
        self.pointer_was_suspended = True

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
                pointer_active = now >= self.pointer_suspended_until
                if pointer_active and self.pointer_was_suspended:
                    # Pick up from wherever the touchpad left the pointer.
                    self.pointer_was_suspended = False
                    self._reset_pointer_state()
                hand_seen = False
                if hand:
                    hands = hand.detect_for_video(image, timestamp).hand_landmarks
                    if hands:
                        hand_seen = True
                        self.last_hand_seen = time.monotonic()
                        points = hands[0]
                        index_tip, thumb_tip = points[8], points[4]
                        if pointer_active:
                            sx, sy = self._smooth_pointer("gestures", index_tip.x, index_tip.y, 5.0)
                            self.event_ready.emit(InteractionEvent("gestures", "pointer", "move_pointer", 0.86, {
                                "x": int(sx * (self.width - 1)), "y": int(sy * (self.height - 1))}))
                        palm_size = math.hypot(points[0].x - points[9].x, points[0].y - points[9].y)
                        pinch_ratio = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y) / max(palm_size, 0.001)
                        now = time.monotonic()
                        if not self.previous_pinch and pinch_ratio <= 0.32:
                            if self.pinch_started_at is None:
                                self.pinch_started_at = now
                            elif now - self.pinch_started_at >= 0.08 and now - self.last_pinch_click >= 0.70:
                                if pointer_active:
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
                                    gaze_active = True
                                    if pointer_active:
                                        sx, sy = self._smooth_pointer(
                                            "gaze", raw[0], raw[1], 6.0, absolute=True)
                                        self.event_ready.emit(InteractionEvent("gaze", "gaze", "move_pointer", 0.82, {
                                            "x": int(sx * (self.width - 1)), "y": int(sy * (self.height - 1))}))
                        if self.head_enabled and not gaze_active:
                            if self.recenter_requested:
                                self.head_tracker.reset()
                                self._forget_source("head")
                                self.recenter_requested = False
                            head_feat = self._extract_head_feature(landmarks)
                            if head_feat is not None:
                                anchor = None
                                if self.head_tracker.smoothed is None:
                                    anchor = self._screen_pointer() or self.pointer_last or (0.5, 0.5)
                                point = self._head_pointer(head_feat, anchor=anchor)
                                if point is not None and pointer_active:
                                    self.pointer_source = "head"
                                    self.pointer_last = point
                                    self.event_ready.emit(InteractionEvent("head", "head", "move_pointer", 0.75, {
                                        "x": int(point[0] * (self.width - 1)), "y": int(point[1] * (self.height - 1))}))
                    elif now - last_face_seen > 3.0 and now - last_face_warning > 10.0:
                        self.status.emit("No face detected - check camera framing and lighting")
                        last_face_warning = now
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

