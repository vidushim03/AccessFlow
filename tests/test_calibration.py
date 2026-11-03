"""Tests for gaze calibration acceptance and saved-mapping validation.

Builds the real window offscreen and drives the actual calibration code with
synthetic data, so a regression shows up without a person in front of the
camera. Run with::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

import main as m
from accessflow.inputs.mapping import (
    GAZE_MODEL_QUADRATIC, apply_gaze_mapping,
)

TARGETS = [(0.12, 0.12), (0.5, 0.12), (0.88, 0.12), (0.12, 0.5),
           (0.5, 0.5), (0.88, 0.5), (0.12, 0.88), (0.5, 0.88), (0.88, 0.88)]

def clean_points():
    """Features and the exact screen targets the calibration wizard uses.

    The feature is an affine function of the screen point, which the
    quadratic model can represent exactly, so a clean fit must reproduce
    every target without clipping.
    """
    features = [(0.18 + 0.62 * tx, 0.22 + 0.55 * ty) for tx, ty in TARGETS]
    return features, list(TARGETS)


class FakeOverlay:
    targets = TARGETS

    def __init__(self):
        self.target_index = 0
        self.updates = 0

    def update(self):
        self.updates += 1

    def close(self):
        pass


class FakeLabel:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = m.AccessFlowWindow()
        cls.w._stop_global_listeners()      # no global hooks during tests
        cls.w.settings = QSettings(QSettings.Format.IniFormat,
                                   QSettings.Scope.UserScope, "AccessFlowTests", "tests")

    @classmethod
    def tearDownClass(cls):
        cls.w.action_executor.stop()
        cls.w.tray.hide()
        cls.w.settings.clear()
        cls.w.deleteLater()

    def setUp(self):
        self.w.settings.clear()
        self.w.gaze_transform = None
        self.w.calibration_notice = ""
        self.w.calibration_points = {}
        self.w.calibration_buffer.clear()
        self.w.calibration_target_index = 0
        self.w.calibration_status = FakeLabel()
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        self.w.calibration_timer.stop()
        self.w.calibration_overlay = None

    def start_target(self):
        self.w.calibration_overlay = FakeOverlay()
        self.w.calibration_timer.start(100)
        self.w.calibration_target_started = time.monotonic()
        return self.w.calibration_overlay

    def test_unstable_point_is_rejected(self):
        self.start_target()
        for _ in range(20):
            self.w.calibration_buffer.append(
                (float(self.rng.uniform(0, 1)), float(self.rng.uniform(0, 1))))
        self.w._poll_calibration()
        self.assertNotIn(0, self.w.calibration_points)
        self.assertEqual(self.w.calibration_target_index, 0)

    def test_keyboard_status_reflects_paused_input_session(self):
        self.w.running = False
        self.w.engine.paused = True
        self.w._show_accessflow_keyboard()
        self.assertEqual(
            self.w.status_label.text(),
            "Keyboard open · start an input session before typing",
        )
        self.w.accessflow_keyboard.hide()

    def test_keyboard_ready_status_requires_active_unpaused_session(self):
        self.w.running = True
        self.w.engine.paused = False
        self.w._show_accessflow_keyboard()
        self.assertEqual(
            self.w.status_label.text(),
            "AccessFlow keyboard ready · select a text field, then a key",
        )
        self.w.accessflow_keyboard.hide()
        self.w.calibration_overlay = None

    def test_stable_point_is_accepted_and_advances(self):
        overlay = self.start_target()
        self.w.calibration_buffer.extend([(0.42, 0.31)] * m.CALIBRATION_MIN_SAMPLES)
        for _ in range(20):
            self.w.calibration_buffer.append(
                (0.42 + float(self.rng.normal(0, 0.002)),
                 0.31 + float(self.rng.normal(0, 0.002))))
        self.w._poll_calibration()
        self.assertNotIn(0, self.w.calibration_points)
        self.w.calibration_target_started -= m.CALIBRATION_MIN_DWELL
        self.w._poll_calibration()
        self.assertIn(0, self.w.calibration_points)
        self.assertEqual(self.w.calibration_target_index, 1)
        self.assertEqual(overlay.updates, 1)

    def test_realistic_still_eye_jitter_is_accepted(self):
        # A still eye still has microsaccadic/landmark jitter (~0.05 std of the
        # iris-in-eye feature). At the 0.07 tolerance this must be accepted so a
        # genuinely still user can finish the nine-point calibration instead of
        # timing out on the first dot.
        overlay = self.start_target()
        self.w.calibration_target_started -= m.CALIBRATION_MIN_DWELL
        self.w.calibration_buffer.extend([(0.42, 0.31)] * m.CALIBRATION_MIN_SAMPLES)
        for _ in range(20):
            self.w.calibration_buffer.append(
                (0.42 + float(self.rng.normal(0, 0.05)),
                 0.31 + float(self.rng.normal(0, 0.05))))
        self.w._poll_calibration()
        self.assertIn(0, self.w.calibration_points)
        self.assertEqual(self.w.calibration_target_index, 1)

    def test_unsteady_eye_is_still_rejected_after_loosening(self):
        # Deliberate eye movement between targets moves the feature by ~0.2+,
        # so it exceeds the research-aligned 0.07 stability tolerance.
        self.start_target()
        for _ in range(20):
            self.w.calibration_buffer.append(
                (float(self.rng.normal(0.42, 0.15)),
                 float(self.rng.normal(0.31, 0.15))))
        self.w._poll_calibration()
        self.assertNotIn(0, self.w.calibration_points)
        self.assertEqual(self.w.calibration_target_index, 0)

    def test_transition_frames_are_discarded(self):
        self.start_target()
        self.w.calibration_buffer.clear()
        self.w.calibration_target_started = time.monotonic()
        self.w._collect_gaze_sample((0.9, 0.9))
        self.assertEqual(len(self.w.calibration_buffer), 0)

    def test_starvation_times_out_instead_of_stalling(self):
        self.start_target()
        self.w.calibration_buffer.clear()
        self.w.calibration_target_started = time.monotonic() - m.CALIBRATION_POINT_TIMEOUT - 1
        self.w._poll_calibration()
        self.assertIn("not enough valid eye readings", self.w.calibration_status.text)
        self.assertIsNone(self.w.calibration_overlay)

    def test_clean_fit_is_saved_with_screen_size(self):
        features, targets = clean_points()
        self.start_target()
        self.w.calibration_points = {i: f for i, f in enumerate(features)}
        self.w._finish_calibration()

        self.assertIsNotNone(self.w.gaze_transform)
        matrix = np.asarray(self.w.gaze_transform)
        self.assertEqual(matrix.shape, (3, 2))
        for (fx, fy), target in zip(features, targets):
            mapped = apply_gaze_mapping(matrix, fx, fy)
            self.assertTrue(np.allclose(mapped, target, atol=1e-6))
        saved = json.loads(self.w.settings.value(m.CALIBRATION_KEY, ""))
        self.assertEqual(saved["feature"], m.GAZE_FEATURE)
        self.assertEqual(saved["model"], m.GAZE_MODEL_AFFINE)
        self.assertEqual(saved["screen"], list(self.w.actions.screen_size()))

    def test_failed_recalibration_keeps_the_working_one(self):
        features, _ = clean_points()
        self.start_target()
        self.w.calibration_points = {i: f for i, f in enumerate(features)}
        self.w._finish_calibration()
        good = self.w.gaze_transform
        saved = self.w.settings.value(m.CALIBRATION_KEY, "")
        self.assertIsNotNone(good)

        self.start_target()
        self.w.calibration_points = {
            i: (float(self.rng.uniform(0, 1)), float(self.rng.uniform(0, 1)))
            for i in range(9)}
        self.w._finish_calibration()

        self.assertEqual(self.w.gaze_transform, good)
        self.assertEqual(self.w.settings.value(m.CALIBRATION_KEY, ""), saved)
        self.assertIn("previous calibration was kept", self.w.calibration_status.text)

    def test_missing_point_aborts_without_saving(self):
        self.start_target()
        self.w.calibration_points = {0: (0.1, 0.1)}
        self.w._finish_calibration()
        self.assertIsNone(self.w.gaze_transform)
        self.assertEqual(self.w.settings.value(m.CALIBRATION_KEY, ""), "")

    def test_mapping_for_another_screen_is_refused(self):
        self.w.settings.setValue(m.CALIBRATION_KEY, json.dumps({
            "feature": m.GAZE_FEATURE,
            "screen": [9999, 9999],
            "matrix": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        }))
        self.assertIsNone(self.w._load_calibration())
        self.assertIn("different screen size", self.w.calibration_notice)

    def test_mapping_for_an_older_feature_is_refused(self):
        self.w.settings.setValue(m.CALIBRATION_KEY, json.dumps({
            "feature": "old_feature",
            "matrix": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        }))
        self.assertIsNone(self.w._load_calibration())
        self.assertIn("older gaze feature", self.w.calibration_notice)

    def test_current_mapping_loads(self):
        self.w.settings.setValue(m.CALIBRATION_KEY, json.dumps({
            "feature": m.GAZE_FEATURE,
            "screen": list(self.w.actions.screen_size()),
            "matrix": [[0.5, 0.1], [0.2, 0.4], [0.1, 0.1]],
        }))
        loaded = self.w._load_calibration()
        self.assertEqual(loaded, [[0.5, 0.1], [0.2, 0.4], [0.1, 0.1]])

    def test_mapping_with_model_shape_mismatch_is_refused(self):
        self.w.settings.setValue(m.CALIBRATION_KEY, json.dumps({
            "feature": m.GAZE_FEATURE,
            "model": GAZE_MODEL_QUADRATIC,
            "screen": list(self.w.actions.screen_size()),
            "matrix": [[0.5, 0.1], [0.2, 0.4], [0.1, 0.1]],
        }))
        self.assertIsNone(self.w._load_calibration())
        self.assertIn("inconsistent", self.w.calibration_notice)

    def test_legacy_mapping_still_loads_with_a_notice(self):
        self.w.settings.setValue("gaze/transform_v3", json.dumps(
            [[0.5, 0.1], [0.2, 0.4], [0.1, 0.1]]))
        loaded = self.w._load_calibration()
        self.assertEqual(loaded, [[0.5, 0.1], [0.2, 0.4], [0.1, 0.1]])
        self.assertIn("Recalibrate", self.w.calibration_notice)

    def test_corrupt_mapping_is_ignored(self):
        self.w.settings.setValue(m.CALIBRATION_KEY, "{not json")
        self.assertIsNone(self.w._load_calibration())

    def test_wrong_shaped_mapping_is_refused(self):
        self.w.settings.setValue(m.CALIBRATION_KEY, json.dumps({
            "feature": m.GAZE_FEATURE,
            "screen": list(self.w.actions.screen_size()),
            "matrix": [[1.0, 0.0], [0.0, 1.0]],
        }))
        self.assertIsNone(self.w._load_calibration())
        self.assertIn("unreadable", self.w.calibration_notice)

    def test_non_finite_mapping_is_refused(self):
        # json round-trips NaN, which numpy accepts but would poison every frame.
        self.w.settings.setValue(m.CALIBRATION_KEY, json.dumps({
            "feature": m.GAZE_FEATURE,
            "screen": list(self.w.actions.screen_size()),
            "matrix": [[float("nan"), 0.0], [0.0, 1.0], [0.0, 0.0]],
        }))
        self.assertIsNone(self.w._load_calibration())
        self.assertIn("unreadable", self.w.calibration_notice)


class GazeTransformTests(unittest.TestCase):
    """A malformed mapping must disable gaze, never stop the camera thread."""

    def setUp(self):
        from accessflow.inputs.vision import VisionWorker
        self.worker = VisionWorker

    def test_bad_shapes_are_rejected(self):
        self.assertIsNone(self.worker._checked_transform([[1.0], [2.0], [3.0]]))
        self.assertIsNone(self.worker._checked_transform("nope"))
        self.assertIsNone(self.worker._checked_transform(None))

    def test_non_finite_values_are_rejected(self):
        self.assertIsNone(self.worker._checked_transform(
            [[float("nan"), 0.0], [0.0, 1.0], [0.0, 0.0]]))

    def test_valid_mapping_is_kept(self):
        kept = self.worker._checked_transform([[0.5, 0.1], [0.2, 0.4], [0.1, 0.1]])
        self.assertEqual(kept.shape, (3, 2))

    def test_quadratic_mapping_is_kept(self):
        kept = self.worker._checked_transform(
            [[0.1, 0.1], [0.6, 0.0], [0.0, 0.5],
             [0.1, 0.0], [0.1, 0.1], [0.0, 0.05]])
        self.assertEqual(kept.shape, (6, 2))


if __name__ == "__main__":
    unittest.main()
