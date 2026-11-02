"""Pointer mapping primitives shared by the vision worker and the UI.

Two very different mappings are used deliberately, because they match how the
body part actually behaves:

* **Hands are relative.** A fingertip has no meaningful absolute screen
  location, so it is treated like a laptop trackpad: the finger's *delta*
  between frames moves the pointer. A dead zone absorbs tremor, and losing
  the hand freezes the pointer and re-anchors on reacquisition.
* **Eyes are absolute.** Gaze is calibrated to the screen with a quadratic
  fit (the same approach as the Eye-To-Mouse reference), because gaze has a
  real, repeatable screen location.

Keeping these here rather than in the camera thread makes them testable
without a webcam.
"""
from __future__ import annotations

import math

import numpy as np

# Hand pointer tuning, ported from the reference gesture_os project. The
# numbers were hand-tuned there: 2.5x finger travel feels like a trackpad,
# 0.008 of the frame (~5 px at 640 wide) is below tremor, and 300 px per
# frame stops a landmark glitch from throwing the pointer across the screen.
HAND_SENSITIVITY = 2.5
HAND_DEAD_ZONE = 0.008
HAND_MAX_DELTA_PX = 300.0
# While the finger is inside the dead zone the reference drifts toward it by
# this fraction each frame. Sustained slow movement still builds up and moves
# the pointer, but random landmark jitter does not accumulate into drift.
HAND_DEAD_ZONE_LEAK = 0.15
# A light EMA on the fingertip halves landmark jitter before the dead zone sees
# it. It is deliberately not the One Euro filter: that filter's slow minimum
# cutoff lagged behind the finger and swallowed small deliberate movements.
# A plain EMA has no such threshold, and its steady-state lag under constant
# motion is only (1-alpha)/alpha of one frame's travel.
HAND_EMA_ALPHA = 0.5

# Gaze model versions. The affine model is the shape saved by earlier
# versions; it is still accepted so an existing calibration is not thrown
# away. New calibrations always use the quadratic model.
GAZE_MODEL_AFFINE = "affine_v1"
GAZE_MODEL_QUADRATIC = "quadratic_v1"
GAZE_QUADRATIC_FEATURES = 6
GAZE_AFFINE_FEATURES = 3


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


class RelativePointer:
    """Trackpad-style pointer driven by fingertip deltas.

    The pointer never follows the finger's absolute position, so moving the
    hand back into frame after a break does not teleport the cursor. It only
    responds to movement *while the hand is visible*.
    """

    def __init__(self, width, height, sensitivity=HAND_SENSITIVITY,
                 dead_zone=HAND_DEAD_ZONE, max_delta_px=HAND_MAX_DELTA_PX,
                 dead_zone_leak=HAND_DEAD_ZONE_LEAK,
                 smoothing=HAND_EMA_ALPHA):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.max_delta_px = max_delta_px
        self.dead_zone_leak = dead_zone_leak
        self.smoothing = max(0.0, min(1.0, smoothing))
        self._x = 0.5
        self._y = 0.5
        self._ref_x = None
        self._ref_y = None
        self._smooth_x = None
        self._smooth_y = None

    @property
    def position(self):
        return (self._x, self._y)

    def _clear_reference(self):
        self._ref_x = None
        self._ref_y = None
        self._smooth_x = None
        self._smooth_y = None

    def reset(self, position=None):
        """Forget the finger reference; optionally move the pointer.

        Used when the touchpad takes over, when the session resumes, or when
        the hand leaves the frame. The next sample only re-anchors.
        """
        self._clear_reference()
        if position is not None:
            self._x = clamp01(position[0])
            self._y = clamp01(position[1])

    def on_hand_lost(self):
        """Freeze the pointer and re-anchor on the next detection."""
        self._clear_reference()

    def update(self, x, y):
        """Return the new normalised screen position for one fingertip sample."""
        x = float(x)
        y = float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return (self._x, self._y)
        if self._smooth_x is None:
            self._smooth_x = x
            self._smooth_y = y
        else:
            alpha = self.smoothing
            self._smooth_x = alpha * x + (1.0 - alpha) * self._smooth_x
            self._smooth_y = alpha * y + (1.0 - alpha) * self._smooth_y
        x, y = self._smooth_x, self._smooth_y

        if self._ref_x is None:
            # First sample after acquisition: remember where the finger is and
            # do not move. This is what prevents a jump when the hand returns.
            self._ref_x = x
            self._ref_y = y
            return (self._x, self._y)

        dx = float(x) - self._ref_x
        dy = float(y) - self._ref_y
        distance = math.hypot(dx, dy)
        if distance < self.dead_zone:
            # Inside the dead zone: do not move, but let the reference follow
            # slowly. A constant slow motion therefore still accumulates,
            # while zero-mean jitter is damped instead of random-walking.
            self._ref_x += dx * self.dead_zone_leak
            self._ref_y += dy * self.dead_zone_leak
            return (self._x, self._y)

        # Past the dead zone the finger's full displacement is applied. The
        # reference only resets on real movement, so sustained slow motion is
        # not swallowed the way a per-frame dead zone would swallow it.
        self._ref_x = float(x)
        self._ref_y = float(y)

        dx_px = dx * self.width * self.sensitivity
        dy_px = dy * self.height * self.sensitivity
        limit = self.max_delta_px
        dx_px = max(-limit, min(dx_px, limit))
        dy_px = max(-limit, min(dy_px, limit))

        self._x = clamp01(self._x + dx_px / self.width)
        self._y = clamp01(self._y + dy_px / self.height)
        return (self._x, self._y)


def gaze_vector(x, y, model=GAZE_MODEL_QUADRATIC):
    """Feature row for the gaze regression.

    The quadratic row mirrors Eye-To-Mouse: a constant, the two iris offsets,
    their interaction, and their squares. It bends the mapping so gaze near
    the screen corners is not systematically short.
    """
    if model == GAZE_MODEL_AFFINE:
        return np.array([1.0, x, y], dtype=float)
    return np.array([1.0, x, y, x * y, x * x, y * y], dtype=float)


def fit_gaze_mapping(features, targets, model=GAZE_MODEL_QUADRATIC):
    """Fit iris features to normalised screen targets.

    Args:
        features: sequence of (iris_x, iris_y) samples, one per target.
        targets: sequence of (screen_x, screen_y) normalised targets.
        model: ``quadratic_v1`` (default) or ``affine_v1`` for tests/legacy.

    Returns:
        ``(matrix, rmse)`` where matrix has shape (6, 2) or (3, 2) and rmse is
        the root-mean-square error in normalised screen units.

    Raises:
        ValueError: if the fit is rank-deficient (the eye barely moved), which
            would otherwise produce a mapping that amplifies noise.
    """
    feature_rows = np.asarray(features, dtype=float)
    target_rows = np.asarray(targets, dtype=float)
    if feature_rows.shape[0] != target_rows.shape[0] or feature_rows.shape[0] < 3:
        raise ValueError("Not enough calibration points.")
    design = np.stack([gaze_vector(fx, fy, model) for fx, fy in feature_rows])
    matrix, _, rank, _ = np.linalg.lstsq(design, target_rows, rcond=None)
    expected_rank = design.shape[1]
    if rank < expected_rank:
        raise ValueError(
            "Eye movement was too small to calibrate. Keep the camera at eye "
            "level and try again.")

    # A quadratic has six coefficients per axis, so nine targets can fit any
    # nonsense perfectly. Report leave-one-out error instead: each point is
    # predicted by a mapping fitted on the other eight. A real calibration
    # generalises; a user who just looked around randomly does not.
    squared_errors = []
    for index in range(len(feature_rows)):
        keep = np.arange(len(feature_rows)) != index
        fold_design = design[keep]
        if np.linalg.matrix_rank(fold_design) < expected_rank:
            continue
        fold_matrix, _, _, _ = np.linalg.lstsq(
            fold_design, target_rows[keep], rcond=None)
        residual = design[index] @ fold_matrix - target_rows[index]
        squared_errors.append(float(np.sum(residual * residual)))
    if squared_errors:
        error = float(np.sqrt(np.mean(squared_errors)))
    else:
        error = float(np.sqrt(np.mean((design @ matrix - target_rows) ** 2)))
    return matrix, error


def apply_gaze_mapping(matrix, x, y):
    """Map one iris feature through a saved mapping, clipped to the screen.

    Accepts both the legacy (3, 2) affine matrix and the current (6, 2)
    quadratic matrix so an old calibration keeps working.
    """
    array = np.asarray(matrix, dtype=float)
    if array.shape == (GAZE_QUADRATIC_FEATURES, 2):
        row = gaze_vector(x, y, GAZE_MODEL_QUADRATIC)
    elif array.shape == (GAZE_AFFINE_FEATURES, 2):
        row = gaze_vector(x, y, GAZE_MODEL_AFFINE)
    else:
        raise ValueError(f"Unsupported gaze mapping shape {array.shape}")
    mapped = row @ array
    return clamp01(mapped[0]), clamp01(mapped[1])


def valid_gaze_matrix(value):
    """Return a finite (3, 2) or (6, 2) mapping, or None.

    A saved mapping can be edited, truncated or written by another version. A
    malformed one raises inside the camera thread and takes the whole vision
    session down, so it is validated before it is ever used.
    """
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape not in ((GAZE_AFFINE_FEATURES, 2), (GAZE_QUADRATIC_FEATURES, 2)):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def model_for_matrix(matrix):
    """Name the model a matrix shape represents."""
    array = np.asarray(matrix)
    if array.shape == (GAZE_QUADRATIC_FEATURES, 2):
        return GAZE_MODEL_QUADRATIC
    return GAZE_MODEL_AFFINE
