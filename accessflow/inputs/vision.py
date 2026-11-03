import ctypes
import logging
import math
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from PySide6.QtCore import QThread, Signal
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, HandLandmarker, HandLandmarkerOptions, RunningMode
from accessflow.core.events import InteractionEvent
from accessflow.inputs.filters import OneEuroFilter2D
from accessflow.inputs.mapping import (
    DwellSelector, RelativePointer, apply_gaze_mapping, valid_gaze_matrix,
)

if sys.platform == "win32":
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
    _user32.GetCursorPos.restype = wintypes.BOOL
else:
    _user32 = None

FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
logger = logging.getLogger(__name__)

CAMERA_BACKENDS = (
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
    ("automatic", cv2.CAP_ANY),
)


class CameraUnavailableError(RuntimeError):
    """Raised when no supported camera backend can produce a frame."""


def open_camera():
    """Find the first usable camera using common Windows backends.

    ``isOpened`` can succeed even when a device does not deliver frames, so a
    candidate is only accepted after one successful read. Failed captures are
    released before trying the next backend/device.
    """
    attempts = []
    for device_index in (0, 1):
        for backend_name, backend in CAMERA_BACKENDS:
            capture = None
            try:
                capture = cv2.VideoCapture(device_index, backend)
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        return capture, f"Camera {device_index} ({backend_name})"
                    attempts.append(f"{device_index}/{backend_name}: no frame")
                else:
                    attempts.append(f"{device_index}/{backend_name}: not opened")
            except Exception as exc:
                attempts.append(f"{device_index}/{backend_name}: {exc}")
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
    tried = ", ".join(attempts)
    logger.warning("Camera unavailable after these attempts: %s", tried)
    raise CameraUnavailableError(
        "Camera unavailable. Tried devices 0 and 1. Close another app using "
        "the camera or check Windows camera permissions, then retry.")


class UIATargetMonitor:
    """Poll the Windows control under the pointer away from the camera loop."""

    def __init__(self, interval=0.12):
        self.interval = max(0.08, float(interval))
        self._lock = threading.Lock()
        self._point = None
        self._target = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="AccessFlowTargetMonitor", daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def set_point(self, point):
        with self._lock:
            self._point = (int(point[0]), int(point[1]))

    def target(self):
        with self._lock:
            return self._target

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.35)

    def _run(self):
        from accessflow.system.actions import WindowsActions

        while not self._stop_event.wait(self.interval):
            with self._lock:
                point = self._point
            if point is None:
                target = None
            else:
                target, _ = WindowsActions.inspect_target(*point)
            with self._lock:
                # Do not publish a hit-test for a point the pointer has
                # already left; stale results can spuriously reset dwell.
                current = self._point
                still_over_target = bool(
                    target is not None and current is not None
                    and len(target) >= 10
                    and target[6] <= current[0] < target[8]
                    and target[7] <= current[1] < target[9])
                if current == point or still_over_target:
                    self._target = target
                elif self._target is not None:
                    # A no-control result for an older pointer position must
                    # not leave the previous control cached indefinitely. Keep
                    # it only while the pointer is still inside that control's
                    # bounds; the next poll will identify the new location.
                    old = self._target
                    still_over_previous = bool(
                        len(old) >= 10
                        and old[6] <= current[0] < old[8]
                        and old[7] <= current[1] < old[9])
                    if not still_over_previous:
                        self._target = None

# Head tracking mirrors the reference tracker in vision-access-hci: the raw
# nose-tip position is compared with a learned neutral pose and scaled.
# The value is a fraction of the screen, so 0.1 of head travel moves the
# pointer a tenth of the screen. Horizontal gain is lower than vertical
# because lateral head rotation produces smaller landmark displacement.
HEAD_X_GAIN = 1.8
HEAD_Y_GAIN = 2.4
# Average the first few samples so a single noisy first frame cannot become the
# permanent neutral pose. The window is restarted whenever the samples move too
# far apart, so a user who is still settling into place never freezes a moving
# pose as "neutral".
HEAD_NEUTRAL_SAMPLES = 12
# Tolerate normal landmark jitter while still rejecting a moving head.
# 3 sigma of a 0.003 jitter window is ~0.009; 0.015 is a safe bound that
# lets a still head settle but restarts if the head is actually moving.
HEAD_NEUTRAL_SPREAD = 0.015
# A single frame that claims the pointer should teleport across more than this
# fraction of the screen diagonal is a landmark glitch, not real motion.
POINTER_JUMP_LIMIT = 0.35
# EMA alpha for head tracking, matching vision-access-hci's 0.18 per frame.
# The reference uses a plain EMA on absolute position with no deadzone and no
# per-frame step clamp; that heavy low-pass is what keeps a still head from
# dragging the pointer.
HEAD_EMA_ALPHA = 0.18
HAND_DWELL_STABILITY_RADIUS = 0.018
HAND_FINGER_EXTEND_ENTER_COS = 0.45
HAND_FINGER_EXTEND_EXIT_COS = 0.20

# Standard MediaPipe hand topology, used by the tracking preview to draw the
# skeleton. Only the preview needs it; detection itself only returns points.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
# Face points worth drawing in the preview: nose tip, both eye corners and
# both iris centres. Drawing all 478 landmarks is just noise at this size.
PREVIEW_FACE_POINTS = (1, 33, 133, 159, 145, 362, 263, 386, 374, 468, 473)
# The preview is a diagnostic view, not the control path, so it can afford to
# be a little behind the live camera.
PREVIEW_INTERVAL = 0.05


class HeadTracker:
    """Head-pointer controller mirroring vision-access-hci.

    The raw nose-tip landmark is compared with a learned neutral pose and
    scaled by the configured gains to produce an absolute target position,
    which is then passed through a plain EMA. There is no per-frame step clamp
    and no deadzone: the EMA alone is the low-pass, and it settles to the
    anchor when the head is still, so a still user produces no pointer drift.

    The neutral pose is learned from a short window of stable samples while
    the pointer is held at the requested anchor, so a bad first detection or
    the user sitting down cannot permanently bias the cursor. Afterwards the
    tracker can be re-anchored to an arbitrary pointer position (used after a
    touchpad takeover or input-method switch) by re-deriving the neutral pose
    from the current feature.
    """

    def __init__(self, alpha=HEAD_EMA_ALPHA, gain_x=HEAD_X_GAIN,
                 gain_y=HEAD_Y_GAIN, neutral_samples=HEAD_NEUTRAL_SAMPLES,
                 neutral_spread=HEAD_NEUTRAL_SPREAD):
        self.alpha = alpha
        self.gain_x = gain_x
        self.gain_y = gain_y
        self.neutral_samples = neutral_samples
        self.neutral_spread = neutral_spread
        self.origin = None
        self._samples = []
        self.smoothed = None
        self.activated = False

    def reset(self):
        """Forget the neutral pose and start learning it again."""
        self.origin = None
        self._samples = []
        self.smoothed = None
        self.activated = False

    def _stable_origin(self, feature):
        """Return a stable neutral feature, or None while still gathering.

        A single bad frame must not become the neutral pose, so the origin is
        the mean of several samples. A window whose samples are spread further
        apart than ``neutral_spread`` is discarded and restarted: that is the
        user still moving into position, not a neutral pose.
        """
        self._samples.append(feature)
        if len(self._samples) < self.neutral_samples:
            return None
        count = len(self._samples)
        median_x = sorted(s[0] for s in self._samples)[count // 2]
        median_y = sorted(s[1] for s in self._samples)[count // 2]
        spread = max(math.hypot(s[0] - median_x, s[1] - median_y)
                     for s in self._samples)
        if spread > self.neutral_spread:
            self._samples = [feature]
            return None
        origin = (sum(s[0] for s in self._samples) / count,
                  sum(s[1] for s in self._samples) / count)
        self._samples = []
        return origin

    def _anchor_to(self, feature, anchor):
        """Derive the neutral pose that puts ``feature`` at ``anchor``."""
        self.origin = (feature[0] - (anchor[0] - 0.5) / self.gain_x,
                       feature[1] - (anchor[1] - 0.5) / self.gain_y)
        self.smoothed = (max(0.0, min(1.0, anchor[0])),
                         max(0.0, min(1.0, anchor[1])))
        self._samples = []

    def update(self, feature, anchor=None):
        """Map a nose-tip feature to a smoothed pointer position.

        Mirrors the reference tracker in vision-access-hci: a plain EMA
        (alpha=0.18) on the *absolute* target position, with no deadzone and
        no per-frame step clamp. That heavy low-pass is what keeps a still
        head from dragging the pointer.

        Args:
            feature: (x, y) raw nose-tip landmark in normalised image space.
            anchor: Optional (x, y) normalised screen position to anchor the
                tracker to. On first activation the neutral pose is learned
                while the pointer is held at this position; afterwards an
                anchor means "resume from here" (a recenter or a switch to
                head control) and is applied immediately.

        Returns:
            (x, y) normalised screen position, or None while learning the
            neutral pose.
        """
        # Learn the neutral pose from a stable 8-sample window while the
        # pointer is held at the requested anchor. A stable window keeps a
        # single bad frame from permanently biasing the neutral pose.
        if anchor is not None:
            if not self.activated:
                origin = self._stable_origin(feature)
                if origin is None:
                    return None
                self.origin = (
                    origin[0] - (anchor[0] - 0.5) / self.gain_x,
                    origin[1] - (anchor[1] - 0.5) / self.gain_y)
                self.smoothed = (max(0.0, min(1.0, anchor[0])),
                                 max(0.0, min(1.0, anchor[1])))
                self.activated = True
                return self.smoothed
            # Takeover/recenter: adopt the current pose as the new neutral.
            self._anchor_to(feature, anchor)
            return self.smoothed
        # If the head tracker has not learned its neutral pose yet, do so
        # before emitting anything so the first frames do not teleport the
        # pointer.
        if self.origin is None:
            origin = self._stable_origin(feature)
            if origin is None:
                return None
            self.origin = origin
            self.activated = True
        # Plain EMA on absolute position (no deadzone, no step clamp), as in
        # vision-access-hci.
        target = (
            max(0.0, min(1.0, (feature[0] - self.origin[0]) * self.gain_x + 0.5)),
            max(0.0, min(1.0, (feature[1] - self.origin[1]) * self.gain_y + 0.5)),
        )
        if self.smoothed is None:
            self.smoothed = target
        else:
            self.smoothed = (
                self.alpha * target[0] + (1.0 - self.alpha) * self.smoothed[0],
                self.alpha * target[1] + (1.0 - self.alpha) * self.smoothed[1],
            )
        return self.smoothed


def _model_path(name, url, stop_requested=None):
    """Fetch a MediaPipe model without leaving a partial asset or hanging.

    Downloads run on the camera worker, so each network read has a timeout and
    the caller can stop between chunks when the user pauses the session.
    """
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AccessFlow" / "models"
    target = base / name
    if not target.exists():
        base.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".download")
        request = urllib.request.Request(
            url, headers={"User-Agent": "AccessFlow/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response, \
                    temporary.open("wb") as output:
                while True:
                    if stop_requested and stop_requested():
                        raise InterruptedError("Camera startup was cancelled")
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if stop_requested and stop_requested():
                raise InterruptedError("Camera startup was cancelled")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
    return str(target)


class VisionWorker(QThread):
    event_ready = Signal(object)
    gaze_sample = Signal(object)
    status = Signal(str)
    # Diagnostic stream for the tracking-preview page. Emitted only when the
    # worker was constructed with ``preview`` set; the payload is a plain dict
    # of a mirrored RGB frame plus landmarks and the mapped cursor.
    preview_ready = Signal(object)
    dwell_feedback = Signal(object)

    def __init__(self, width, height, gaze_enabled, gestures_enabled,
                 gaze_transform=None, head_enabled=False, preview=None,
                 dwell_ms=1200):
        super().__init__()
        self.width, self.height = width, height
        self.gaze_enabled, self.gestures_enabled, self.head_enabled = gaze_enabled, gestures_enabled, head_enabled
        self.preview = preview
        self.stop_requested = False
        self.dwell_ms = max(500, min(3000, int(dwell_ms)))
        self.screen_dwell = DwellSelector(self.dwell_ms)
        self.dwell_started_at = None
        self.dwell_anchor = None
        self.dwell_candidate_at = None
        self.dwell_armed = False
        self.next_dwell_button = "left"
        self.dwell_candidate_button = None
        self.dwell_stability_ms = 250
        self.dwell_pointer_anchor = None
        self.dwell_locked = False
        self.dwell_lock_anchor = None
        self.dwell_cooldown_until = 0.0
        self.hand_dwell_target = None
        self.hand_target_candidate = None
        self.hand_target_candidate_at = None
        self.hand_target_stability_s = 0.20
        self.target_monitor = UIATargetMonitor()
        self.hand_suppressed_until = 0.0
        self.hand_pinched = False
        self.hand_pinch_enter_ratio = 0.32
        self.hand_pinch_exit_ratio = 0.48
        self.hand_finger_extended = {
            "index": False, "middle": False, "ring": False, "pinky": False}
        self.last_hand_seen = 0.0
        self.pointer_suspended_until = 0.0
        self.pointer_was_suspended = False
        self.pointer_source = None
        self.pointer_output = {}
        self.pointer_offset = {}
        self.pointer_last = None
        self.head_tracker = HeadTracker()
        # Hands use a relative trackpad mapping, never an absolute one. The
        # dead zone with a fixed reference does the tremor suppression; a
        # pre-filter would lag behind the finger and swallow slow movement.
        self.hand_pointer = RelativePointer(width, height)
        # Live control should continue from the user's current cursor location,
        # not jump to screen center when the first hand frame arrives.
        if not self.preview:
            self.hand_pointer.reset(self._screen_pointer() or (0.5, 0.5))
        self.recenter_requested = False
        self.pointer_filters = {
            # Eye gaze is inherently noisy and calibrated; keep it slow.
            "gaze": OneEuroFilter2D(0.15, 0.005),
        }
        # Head tracking smooths itself inside HeadTracker, so it has no entry
        # here; a filter left unused would only suggest it still applied.
        self._screen_diagonal = math.hypot(width, height)
        self.gaze_transform = self._checked_transform(gaze_transform)

    @staticmethod
    def _checked_transform(gaze_transform):
        """Accept only a finite affine or quadratic mapping.

        A malformed mapping would raise on every frame and stop the whole
        camera session, which also takes head and hand tracking down with it.
        """
        if gaze_transform is None:
            return None
        return valid_gaze_matrix(gaze_transform)

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
        pointer_filter = self.pointer_filters.get(source)
        if pointer_filter is not None:
            pointer_filter.reset()
        self.pointer_output.pop(source, None)
        self.pointer_offset.pop(source, None)
        if self.pointer_source == source:
            self.pointer_source = None

    def _reset_pointer_state(self):
        """Forget every source, e.g. after the touchpad has moved the pointer."""
        self.pointer_source = None
        self.pointer_output.clear()
        self.pointer_offset.clear()
        # A dwell started before touchpad takeover must never complete after
        # camera control resumes at a different cursor position.
        self.dwell_anchor = None
        self.dwell_started_at = None
        self.dwell_candidate_at = None
        self.dwell_armed = False
        self.dwell_candidate_button = None
        self.dwell_locked = False
        self.dwell_lock_anchor = None
        self.dwell_cooldown_until = 0.0
        self.hand_dwell_target = None
        self.hand_target_candidate = None
        self.hand_target_candidate_at = None
        self.hand_pinched = False
        for finger in self.hand_finger_extended:
            self.hand_finger_extended[finger] = False
        self.screen_dwell.reset()
        # Resume the relative hand pointer where the real cursor is, so the
        # first hand movement after a touchpad takeover continues from there.
        if self.preview:
            self.hand_pointer.reset((0.5, 0.5))
        else:
            self.hand_pointer.reset(self._screen_pointer() or self.pointer_last)
        # The head tracker holds its own smoothed state; drop it so the next
        # head frame re-anchors to wherever the pointer now is.
        self.head_tracker.smoothed = None

    def _smooth_pointer(self, source, x, y, deadzone_px, absolute=True):
        """Filter one absolute (calibrated) pointer sample.

        Only gaze uses this path now; hands are handled by ``RelativePointer``
        and head by ``HeadTracker``. Gaze is calibrated to absolute screen
        coordinates, so it is never anchored to the previous pointer.
        """
        now = time.monotonic()
        if self.pointer_source != source:
            # Never let a filter carry motion across an input-method change.
            self.pointer_filters[source].reset()
            self.pointer_output.pop(source, None)
        filtered = self.pointer_filters[source].update((x, y), now)
        previous = self.pointer_output.get(source)
        if previous is None:
            current = (max(0.0, min(1.0, filtered[0])),
                       max(0.0, min(1.0, filtered[1])))
            self.pointer_offset[source] = (0.0, 0.0)
            self.pointer_output[source] = current
            self.pointer_source = source
            self.pointer_last = current
            return current
        target_x = filtered[0] + self.pointer_offset[source][0]
        target_y = filtered[1] + self.pointer_offset[source][1]
        dx = (target_x - previous[0]) * self.width
        dy = (target_y - previous[1]) * self.height
        if math.hypot(dx, dy) < deadzone_px:
            return previous
        if math.hypot(dx, dy) > self._screen_diagonal * POINTER_JUMP_LIMIT:
            # A bad landmark frame can claim a huge jump. Keep the pointer still
            # and rebase on the new sample so normal movement can continue.
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
            if not all(math.isfinite(value) for value in
                       (iris.x, iris.y, a.x, a.y, b.x, b.y)):
                continue
            axis_x = b.x - a.x
            axis_y = b.y - a.y
            width_sq = axis_x * axis_x + axis_y * axis_y
            if not math.isfinite(width_sq) or width_sq < 1e-6:
                continue
            rel_x = iris.x - a.x
            rel_y = iris.y - a.y
            x = (rel_x * axis_x + rel_y * axis_y) / width_sq
            y = (axis_x * rel_y - axis_y * rel_x) / width_sq
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
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

    def _emit_preview(self, rgb, points, cursor, label, calibrated=True):
        """Publish one diagnostic frame to the tracking-preview page."""
        if not self.preview:
            return
        self.preview_ready.emit({
            "mode": self.preview,
            "frame": rgb,
            "points": points or [],
            "cursor": cursor,
            "label": label,
            "calibrated": calibrated,
            "dwell_progress": self._preview_dwell_progress,
        })

    def _update_screen_dwell(self, source, cursor, now, pointer_active):
        """Update eye/head dwell and emit a one-shot left click when live."""
        point = (cursor[0] * (self.width - 1),
                 cursor[1] * (self.height - 1))
        target = None
        if self.preview is None and pointer_active:
            # Windows UI Automation may block; a helper thread keeps camera
            # tracking responsive while refreshing the control under the cursor.
            self.target_monitor.set_point(point)
            target = self.target_monitor.target()
        progress, clicked = self.screen_dwell.update(
            point, now, source, enabled=(self.preview is not None or pointer_active),
            target=target)
        self._preview_dwell_progress = progress
        if not self.preview and (progress < 1.0 or clicked):
            self.dwell_feedback.emit({"point": point, "progress": progress})
        if clicked and not self.preview:
            self.event_ready.emit(InteractionEvent(
                source, "dwell", "click_target", 0.90, {
                    "button": "left", "x": int(point[0]), "y": int(point[1])}))
        return progress, clicked

    def _hand_tracking(self, points, now, pointer_active):
        """Map one detected hand and update the relative pointer.

        Returns ``(cursor, label)`` for the preview and the live pointer path.
        """
        index_tip = points[8]
        if self.preview == "hand":
            # Preview is for checking the hand-to-pointer path, never for
            # rehearsing a click. In particular, do not arm dwell or freeze
            # the cursor here: preview cannot click, and a frozen cursor made
            # hand tracking look broken after the index-only dwell change.
            cursor = self.hand_pointer.update(index_tip.x, index_tip.y)
            return cursor, "Move index to steer · preview does not click"

        pose = self._hand_pose(points)
        # Keep pinching explicitly out of the dwell path; otherwise a held
        # pinch could cause an unintended click while the fingertip is steady.
        if self._hand_is_pinched(points):
            pose = "other"
        if self.dwell_locked:
            if (self.dwell_lock_anchor is not None
                    and math.hypot(index_tip.x - self.dwell_lock_anchor[0],
                                   index_tip.y - self.dwell_lock_anchor[1]) > 0.03):
                self.dwell_locked = False
                self.dwell_lock_anchor = None
                self.hand_pointer.on_hand_lost()
                cursor = self.hand_pointer.position
                self.dwell_armed = False
                self.dwell_started_at = now
                self.dwell_anchor = (index_tip.x, index_tip.y)
                self.dwell_candidate_at = None
                self.dwell_candidate_button = None
                self.dwell_cooldown_until = 0.0
                self.dwell_candidate_at = now
                self.dwell_candidate_button = (
                    None if self.hand_pinched else
                    "right" if pose == "two_fingers" else "left")
                self.dwell_pointer_anchor = None
            else:
                cursor = self.hand_pointer.position
                return cursor, "Move away to re-arm clicking"
        # Keep mapping the fingertip during dwell. Freezing output made hand
        # control appear to stop as soon as the timer armed. Dwell is canceled
        # below if the fingertip moves beyond its fixed stability anchor.
        cursor = self.hand_pointer.position
        if not self.dwell_locked:
            cursor = self.hand_pointer.update(index_tip.x, index_tip.y)
        target = None
        if pointer_active:
            screen_point = (cursor[0] * (self.width - 1),
                            cursor[1] * (self.height - 1))
            self.target_monitor.set_point(screen_point)
            target = self.target_monitor.target()
        if target == self.hand_dwell_target:
            self.hand_target_candidate = None
            self.hand_target_candidate_at = None
        elif target != self.hand_target_candidate:
            self.hand_target_candidate = target
            self.hand_target_candidate_at = now
        elif (self.hand_target_candidate_at is not None
              and now - self.hand_target_candidate_at >= self.hand_target_stability_s):
            # Hit-testing can alternate between a control and its parent as
            # the pointer jitters. Require the new target to persist briefly
            # before resetting an otherwise stable hand dwell.
            self.hand_dwell_target = target
            self.hand_target_candidate = None
            self.hand_target_candidate_at = None
            self.dwell_armed = False
            self.dwell_started_at = None
            self.dwell_candidate_at = None
            self.dwell_candidate_button = None
            self.dwell_anchor = (index_tip.x, index_tip.y)
        current = (index_tip.x, index_tip.y)
        if self.dwell_anchor is None:
            self.dwell_anchor = current
            self.dwell_started_at = now
        elif math.hypot(current[0] - self.dwell_anchor[0],
                        current[1] - self.dwell_anchor[1]) > HAND_DWELL_STABILITY_RADIUS:
            # Before arming, any movement restarts the stability interval. Once
            # armed, the anchor stays fixed for the full dwell duration; easing
            # it toward the finger let slow motion continue unnoticed and could
            # trigger a click at a location the user had already left.
            self.dwell_anchor = current
            if self.dwell_armed:
                self.dwell_armed = False
                self.dwell_candidate_button = None
                self.dwell_candidate_at = None
                self.dwell_started_at = None
            elif self.dwell_candidate_button is not None:
                self.dwell_candidate_at = now

        # Left click depends only on index-tip stability; the other fingers
        # may be relaxed or curled. Reserve two raised fingers for right click.
        button = None if self.hand_pinched else (
            "right" if pose == "two_fingers" else "left")
        if now < self.dwell_cooldown_until:
            if pointer_active:
                self.dwell_feedback.emit({"point": (cursor[0] * (self.width - 1),
                                                     cursor[1] * (self.height - 1)),
                                          "progress": 0.0})
            self.dwell_armed = False
            self.dwell_started_at = None
            if button is None:
                self.dwell_candidate_button = None
                self.dwell_candidate_at = None
            return cursor, "Click sent - wait briefly"
        if button is not None:
            if button != self.dwell_candidate_button:
                self.dwell_candidate_button = button
                self.dwell_candidate_at = now
                self.dwell_armed = False
            elif (not self.dwell_armed and self.dwell_candidate_at is not None
                  and now - self.dwell_candidate_at >= self.dwell_stability_ms / 1000.0):
                self.dwell_armed = True
                self.next_dwell_button = button
                self.dwell_started_at = now
                self.dwell_anchor = current
            elif self.dwell_armed:
                if math.hypot(current[0] - self.dwell_anchor[0],
                              current[1] - self.dwell_anchor[1]) > HAND_DWELL_STABILITY_RADIUS:
                    self.dwell_armed = False
                    self.dwell_candidate_button = None
                    self.dwell_candidate_at = None
                    self.dwell_started_at = None
                elif now - self.dwell_started_at >= self.dwell_ms / 1000.0:
                    self.dwell_feedback.emit({
                        "point": (cursor[0] * (self.width - 1),
                                  cursor[1] * (self.height - 1)),
                        "progress": 1.0})
                    if not self.preview:
                        self.event_ready.emit(InteractionEvent(
                            "gestures", "dwell", "click_target", 0.90, {
                                "button": button,
                                "x": int(cursor[0] * (self.width - 1)),
                                "y": int(cursor[1] * (self.height - 1))}))
                    self.dwell_cooldown_until = now + 0.45
                    self.dwell_armed = False
                    self.dwell_started_at = None
                    self.dwell_anchor = current
                    self.dwell_locked = True
                    self.dwell_lock_anchor = current
                    return cursor, f"{button.title()} click sent"
            if pointer_active:
                progress = (max(0.0, min(1.0, (now - self.dwell_started_at)
                                        / (self.dwell_ms / 1000.0)))
                            if self.dwell_armed and self.dwell_started_at is not None
                            else 0.0)
                self.dwell_feedback.emit({
                    "point": (cursor[0] * (self.width - 1),
                              cursor[1] * (self.height - 1)),
                    "progress": progress})
        else:
            if pointer_active:
                self.dwell_feedback.emit({"point": (cursor[0] * (self.width - 1),
                                                     cursor[1] * (self.height - 1)),
                                          "progress": 0.0})
            self.dwell_armed = False
            self.dwell_candidate_button = None
            self.dwell_candidate_at = None
            self.dwell_started_at = None
        if self.hand_pinched:
            return cursor, "Pinch ignored - release to use dwell"
        label = (f"Hold still for {self.next_dwell_button} click"
                 if self.dwell_armed else "Move index to steer")
        return cursor, label

    def _hand_pose(self, points):
        """Classify fingers from local joint direction with hysteresis.

        Wrist-to-tip distances changed too much as the hand rotated, making a
        held two-finger pose flicker into the left-click state. Local MCP/PIP/
        tip alignment is rotation-tolerant, and separate enter/exit thresholds
        keep landmark noise from repeatedly changing the right-click pose.
        """
        finger_joints = (
            ("index", 5, 6, 8), ("middle", 9, 10, 12),
            ("ring", 13, 14, 16), ("pinky", 17, 18, 20),
        )
        extended = {}
        for name, mcp_index, pip_index, tip_index in finger_joints:
            mcp, pip, tip = points[mcp_index], points[pip_index], points[tip_index]
            proximal = (pip.x - mcp.x, pip.y - mcp.y)
            distal = (tip.x - pip.x, tip.y - pip.y)
            proximal_length = math.hypot(*proximal)
            distal_length = math.hypot(*distal)
            previous = self.hand_finger_extended[name]
            if proximal_length < 1e-4 or distal_length < 1e-4:
                is_extended = previous
            else:
                alignment = ((proximal[0] * distal[0]
                              + proximal[1] * distal[1])
                             / (proximal_length * distal_length))
                threshold = (HAND_FINGER_EXTEND_EXIT_COS if previous else
                             HAND_FINGER_EXTEND_ENTER_COS)
                is_extended = (alignment >= threshold
                               and distal_length / proximal_length >= 0.45)
            self.hand_finger_extended[name] = is_extended
            extended[name] = is_extended

        index_middle = extended["index"] and extended["middle"]
        if index_middle and not extended["ring"] and not extended["pinky"]:
            return "two_fingers"
        if all(extended.values()):
            return "open"
        return "other"

    def _hand_is_pinched(self, points):
        """Use a hand-size-relative hysteresis gate to suppress pinch poses."""
        palm_size = math.hypot(points[0].x - points[9].x,
                               points[0].y - points[9].y)
        if palm_size < 1e-4:
            self.hand_pinched = False
            return False
        ratio = math.hypot(points[8].x - points[4].x,
                           points[8].y - points[4].y) / palm_size
        if self.hand_pinched:
            if ratio >= self.hand_pinch_exit_ratio:
                self.hand_pinched = False
        elif ratio <= self.hand_pinch_enter_ratio:
            self.hand_pinched = True
        return self.hand_pinched

    def run(self):
        capture = face = hand = None
        reported_stop = False
        try:
            capture, camera_description = open_camera()
            self.target_monitor.start()
            if self.gaze_enabled or self.head_enabled:
                face = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_model_path(
                        "face_landmarker.task", FACE_URL,
                        stop_requested=lambda: self.stop_requested)),
                    running_mode=RunningMode.VIDEO, num_faces=1,
                    min_face_detection_confidence=0.70, min_face_presence_confidence=0.60,
                    min_tracking_confidence=0.65))
            if self.gestures_enabled:
                hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_model_path(
                        "hand_landmarker.task", HAND_URL,
                        stop_requested=lambda: self.stop_requested)),
                    running_mode=RunningMode.VIDEO, num_hands=1, min_hand_detection_confidence=0.70,
                    min_hand_presence_confidence=0.60, min_tracking_confidence=0.65))
            self.status.emit(f"Camera inputs active - {camera_description}")
            timestamp = int(time.monotonic() * 1000)
            last_face_seen = time.monotonic()
            last_face_warning = 0.0
            face_seen = False
            last_brightness_check = 0.0
            last_preview = 0.0
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
                preview_points = []
                preview_cursor = None
                preview_label = ""
                preview_calibrated = True
                self._preview_dwell_progress = 0.0
                if hand:
                    hands = hand.detect_for_video(image, timestamp).hand_landmarks
                    if hands:
                        hand_seen = True
                        self.screen_dwell.reset()
                        self.last_hand_seen = time.monotonic()
                        points = hands[0]
                        if not getattr(self, "_hand_reported", False):
                            self._hand_reported = True
                            self.status.emit("Hand detected")
                        preview_points = [(p.x, p.y) for p in points]
                        if pointer_active:
                            if not self.preview and self.pointer_source != "gestures":
                                # Re-anchor on control handoff (for example
                                # when eye/head control was also selected).
                                self.hand_pointer.reset(
                                    self._screen_pointer() or self.pointer_last
                                    or (0.5, 0.5))
                            cursor, preview_label = self._hand_tracking(
                                points, now, pointer_active)
                            preview_cursor = cursor
                            if not self.preview:
                                self.pointer_source = "gestures"
                                self.pointer_last = cursor
                                if not self.dwell_locked:
                                    self.event_ready.emit(InteractionEvent(
                                        "gestures", "pointer", "move_pointer", 0.86, {
                                            "x": int(cursor[0] * (self.width - 1)),
                                            "y": int(cursor[1] * (self.height - 1))}))
                        else:
                            preview_label = "Pointer handed to touchpad"
                    else:
                        # The hand left the frame: freeze the cursor and
                        # re-anchor on the next detection. This is what stops
                        # the pointer running off when the hand comes back.
                        self.hand_pointer.on_hand_lost()
                        self.dwell_anchor = None
                        self.dwell_started_at = None
                        self.dwell_armed = False
                        self.dwell_candidate_at = None
                        self.dwell_candidate_button = None
                        self.dwell_pointer_anchor = None
                        self.dwell_locked = False
                        self.dwell_lock_anchor = None
                        self.dwell_cooldown_until = 0.0
                        self.hand_pinched = False
                        for finger in self.hand_finger_extended:
                            self.hand_finger_extended[finger] = False
                        self.hand_dwell_target = None
                        self.hand_target_candidate = None
                        self.hand_target_candidate_at = None
                        if self.preview == "hand":
                            preview_label = "Show your hand to the camera"
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
                                dwell_progress = 0.0
                                rx, ry = gaze_feat
                                self.gaze_sample.emit((rx, ry))
                                if self.gaze_transform is not None:
                                    raw = apply_gaze_mapping(self.gaze_transform, rx, ry)
                                    gaze_active = True
                                    sx, sy = self._smooth_pointer(
                                        "gaze", raw[0], raw[1], 6.0, absolute=True)
                                    dwell_progress, _ = self._update_screen_dwell(
                                        "gaze", (sx, sy), now, pointer_active)
                                    if pointer_active and not self.preview:
                                        self.pointer_source = "gaze"
                                        self.pointer_last = (sx, sy)
                                        self.event_ready.emit(InteractionEvent("gaze", "gaze", "move_pointer", 0.82, {
                                            "x": int(sx * (self.width - 1)), "y": int(sy * (self.height - 1))}))
                                    # Preview and live control must show the
                                    # same filtered mapping, so a user can
                                    # diagnose jitter before enabling desktop
                                    # pointer movement.
                                    preview_cursor = (sx, sy)
                                elif self.preview:
                                    # Uncalibrated preview still shows the raw
                                    # iris position so the user can see that
                                    # the eyes are being tracked.
                                    preview_cursor = (max(0.0, min(1.0, rx)),
                                                      max(0.0, min(1.0, ry)))
                                    preview_calibrated = False
                                if self.preview:
                                    preview_points = [
                                        (landmarks[i].x, landmarks[i].y)
                                        for i in PREVIEW_FACE_POINTS]
                                    if preview_calibrated:
                                        preview_label = (
                                            f"Hold still for left click · {dwell_progress:.0%}"
                                            if dwell_progress > 0 else
                                            "Hold gaze steady to dwell-click")
                                    else:
                                        preview_label = "Not calibrated - run the nine-point calibration"
                        if self.head_enabled and not gaze_active:
                            if self.recenter_requested:
                                self.head_tracker.reset()
                                self._forget_source("head")
                                self.recenter_requested = False
                            head_feat = self._extract_head_feature(landmarks)
                            if head_feat is not None:
                                anchor = None
                                if self.head_tracker.smoothed is None:
                                    if self.preview:
                                        anchor = (0.5, 0.5)
                                    else:
                                        anchor = self._screen_pointer() or self.pointer_last or (0.5, 0.5)
                                elif (not self.preview
                                      and self.pointer_source not in (None, "head")):
                                    # Another input method moved the pointer since
                                    # the last head frame: resume from there
                                    # instead of jumping back to the old pose.
                                    anchor = (self._screen_pointer()
                                              or self.pointer_last or (0.5, 0.5))
                                point = self._head_pointer(head_feat, anchor=anchor)
                                if self.preview:
                                    preview_points = [
                                        (landmarks[i].x, landmarks[i].y)
                                        for i in PREVIEW_FACE_POINTS]
                                if point is not None:
                                    preview_cursor = point
                                    dwell_progress, _ = self._update_screen_dwell(
                                        "head", point, now, pointer_active)
                                    if self.preview:
                                        preview_label = (
                                            f"Hold still for left click · {dwell_progress:.0%}"
                                            if dwell_progress > 0 else
                                            "Move your head; hold still to dwell-click")
                                    elif pointer_active:
                                        self.pointer_source = "head"
                                        self.pointer_last = point
                                        self.event_ready.emit(InteractionEvent("head", "head", "move_pointer", 0.75, {
                                            "x": int(point[0] * (self.width - 1)), "y": int(point[1] * (self.height - 1))}))
                                elif self.preview:
                                    preview_label = "Hold still - learning your neutral position"
                    elif now - last_face_seen > 3.0 and now - last_face_warning > 10.0:
                        self.status.emit("No face detected - check camera framing and lighting")
                        last_face_warning = now
                    elif self.preview and self.preview != "hand":
                        preview_label = "Show your face to the camera"
                if self.preview and now - last_preview >= PREVIEW_INTERVAL:
                    last_preview = now
                    self._emit_preview(rgb, preview_points, preview_cursor,
                                       preview_label, preview_calibrated)
        except CameraUnavailableError as exc:
            self.status.emit(str(exc))
            reported_stop = True
        except Exception as exc:
            self.status.emit(f"Camera stopped: {exc}")
            reported_stop = True
        finally:
            self.target_monitor.stop()
            # A MediaPipe close failure must not leave the camera device open;
            # release each resource independently during every exit path.
            for resource, method_name in (
                    (face, "close"), (hand, "close"),
                    (capture, "release")):
                if resource is None:
                    continue
                try:
                    getattr(resource, method_name)()
                except Exception:
                    logger.exception("Could not %s camera resource", method_name)
            if not reported_stop:
                self.status.emit("Camera inputs stopped")

