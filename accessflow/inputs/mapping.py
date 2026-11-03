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

# Keep hand response quick enough to follow slow intentional movement while
# retaining a modest jitter threshold and per-frame speed cap.
HAND_SENSITIVITY = 2.0
HAND_DEAD_ZONE = 0.009
HAND_MAX_DELTA_PX = 220.0
# Keep the reference fixed below the threshold. This prevents small landmark
# jitter from accumulating into cursor motion; intended movement accumulates
# against the reference until it crosses the dead zone.
HAND_DEAD_ZONE_LEAK = 0.0
# A light EMA on the fingertip halves landmark jitter before the dead zone sees
# it. It is deliberately not the One Euro filter: that filter's slow minimum
# cutoff lagged behind the finger and swallowed small deliberate movements.
# A plain EMA has no such threshold, and its steady-state lag under constant
# motion is only (1-alpha)/alpha of one frame's travel.
HAND_EMA_ALPHA = 0.60
SCREEN_DWELL_RADIUS_PX = 30.0

# Gaze model versions. The affine model is the shape saved by earlier
# versions; it is still accepted so an existing calibration is not thrown
# away. New calibrations always use the quadratic model.
GAZE_MODEL_AFFINE = "affine_v1"
GAZE_MODEL_QUADRATIC = "quadratic_v1"
GAZE_QUADRATIC_FEATURES = 6
GAZE_AFFINE_FEATURES = 3


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


class DwellSelector:
    """One-shot left-click dwell for an absolute screen pointer and target.

    When a UI control identity is available, changing controls restarts dwell.
    Otherwise the selector uses a small spatial radius. It fires once and
    rearms only after the pointer leaves the target/radius.
    """

    def __init__(self, duration_ms=1200, radius_px=SCREEN_DWELL_RADIUS_PX,
                 dropout_reset_s=0.25):
        self.duration_ms = max(250, int(duration_ms))
        self.radius_px = max(1.0, float(radius_px))
        self.dropout_reset_s = max(0.0, float(dropout_reset_s))
        self.reset()

    def reset(self):
        self.source = None
        self.target = None
        self.anchor = None
        self.started_at = None
        self.last_update = None
        self.fired = False

    def update(self, point, now, source, enabled=True, target=None):
        """Return ``(progress, clicked)`` for a pixel-space pointer sample."""
        if not enabled:
            self.reset()
            return 0.0, False
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            self.reset()
            return 0.0, False
        now = float(now)
        if (self.source != source
                or (self.last_update is not None
                    and now - self.last_update > self.dropout_reset_s)
                or (self.anchor is not None and target != self.target)):
            self.reset()
        if self.anchor is None:
            self.source = source
            self.target = target
            self.anchor = (x, y)
            self.started_at = now
            self.last_update = now
            return 0.0, False

        if math.hypot(x - self.anchor[0], y - self.anchor[1]) > self.radius_px:
            self.anchor = (x, y)
            self.started_at = now
            self.fired = False
        self.target = target
        self.last_update = now
        if self.fired:
            return 1.0, False
        progress = max(0.0, min(1.0,
                                (now - self.started_at) * 1000.0 / self.duration_ms))
        if progress >= 1.0:
            self.fired = True
            return 1.0, True
        return progress, False


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
            # Inside the dead zone: keep the reference fixed. Slow deliberate
            # movement accumulates against it and eventually crosses the zone;
            # following the reference here turns symmetric landmark noise into
            # a random walk across the screen.
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
    if (feature_rows.ndim != 2 or feature_rows.shape[1:] != (2,)
            or target_rows.ndim != 2 or target_rows.shape[1:] != (2,)):
        raise ValueError("Gaze features and targets must be 2D points.")
    if not (np.all(np.isfinite(feature_rows))
            and np.all(np.isfinite(target_rows))):
        raise ValueError("Gaze calibration contains non-finite readings.")
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


def leave_one_out_gaze_predictions(features, targets, model):
    """Return each held-out target prediction for calibration diagnostics.

    A ``None`` prediction means the remaining calibration points could not
    identify a full-rank model for that fold.
    """
    feature_rows = np.asarray(features, dtype=float)
    target_rows = np.asarray(targets, dtype=float)
    if (feature_rows.ndim != 2 or feature_rows.shape[1:] != (2,)
            or target_rows.ndim != 2 or target_rows.shape[1:] != (2,)):
        raise ValueError("Gaze features and targets must be 2D points.")
    if not (np.all(np.isfinite(feature_rows))
            and np.all(np.isfinite(target_rows))):
        raise ValueError("Gaze calibration contains non-finite readings.")
    if feature_rows.shape[0] != target_rows.shape[0]:
        raise ValueError("Feature and target counts do not match.")
    design = np.stack([gaze_vector(fx, fy, model) for fx, fy in feature_rows])
    predictions = []
    for index in range(len(feature_rows)):
        keep = np.arange(len(feature_rows)) != index
        fold_design = design[keep]
        if np.linalg.matrix_rank(fold_design) < design.shape[1]:
            predictions.append(None)
            continue
        fold_matrix, _, _, _ = np.linalg.lstsq(
            fold_design, target_rows[keep], rcond=None)
        predictions.append(design[index] @ fold_matrix)
    return predictions


def fit_best_gaze_mapping(features, targets, quadratic_improvement=0.005):
    """Choose the simplest gaze model whose held-out fit is adequate.

    Nine calibration points provide much less evidence for a six-parameter
    quadratic than for a three-parameter affine mapping. Start with affine and
    use quadratic only when leave-one-out validation improves by at least
    ``quadratic_improvement`` of normalized screen space. This avoids selecting
    a more flexible model for a negligible change while retaining it when the
    eye response is meaningfully curved.

    Returns ``(matrix, error, model, errors)``; ``errors`` contains the
    cross-validation score for each model that could be fitted.
    """
    errors = {}
    matrices = {}
    for model in (GAZE_MODEL_AFFINE, GAZE_MODEL_QUADRATIC):
        try:
            matrices[model], errors[model] = fit_gaze_mapping(
                features, targets, model)
        except ValueError:
            continue
    if not matrices:
        raise ValueError("Eye movement was too small to fit a gaze mapping.")

    affine_error = errors.get(GAZE_MODEL_AFFINE, float("inf"))
    quadratic_error = errors.get(GAZE_MODEL_QUADRATIC, float("inf"))
    if (GAZE_MODEL_QUADRATIC in matrices
            and affine_error - quadratic_error >= quadratic_improvement):
        chosen = GAZE_MODEL_QUADRATIC
    elif GAZE_MODEL_AFFINE in matrices:
        chosen = GAZE_MODEL_AFFINE
    else:
        chosen = GAZE_MODEL_QUADRATIC
    return matrices[chosen], errors[chosen], chosen, errors


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
