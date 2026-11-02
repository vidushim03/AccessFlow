"""Tests for the pointer mappings that keep the cursor still.

These cover the two complaints that are hardest to see in a screenshot:

* a still hand must not move the pointer, and a hand that leaves and returns
  must not teleport the pointer;
* a still head must settle and stay put.

They also cover the quadratic gaze model and its leave-one-out validation.
Run with::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from accessflow.inputs.mapping import (
    GAZE_MODEL_QUADRATIC, RelativePointer, apply_gaze_mapping,
    fit_gaze_mapping, gaze_vector, valid_gaze_matrix,
)
from accessflow.inputs.vision import HEAD_NEUTRAL_SAMPLES, HeadTracker

TARGETS = [(0.12, 0.12), (0.5, 0.12), (0.88, 0.12), (0.12, 0.5),
           (0.5, 0.5), (0.88, 0.5), (0.12, 0.88), (0.5, 0.88), (0.88, 0.88)]

CURVED_RESPONSE = np.array([
    [0.10, 0.12],
    [0.62, 0.08],
    [0.04, 0.54],
    [0.10, -0.03],
    [0.11, 0.09],
    [-0.02, 0.06],
])


class RelativePointerTests(unittest.TestCase):
    def setUp(self):
        self.pointer = RelativePointer(1920, 1080)

    def test_first_sample_only_anchors(self):
        self.assertEqual(self.pointer.update(0.31, 0.27), (0.5, 0.5))

    def test_still_finger_does_not_drift(self):
        self.pointer.update(0.40, 0.40)
        # Micro-tremor below the dead zone, alternating like real landmark
        # jitter. The pointer must not creep across the screen.
        for index in range(300):
            wobble = 0.0025 if index % 2 else -0.0025
            position = self.pointer.update(0.40 + wobble, 0.40 - wobble)
        self.assertAlmostEqual(position[0], 0.5, places=6)
        self.assertAlmostEqual(position[1], 0.5, places=6)

    def test_movement_moves_the_pointer(self):
        self.pointer.update(0.40, 0.40)
        x, y = self.pointer.update(0.42, 0.40)
        self.assertGreater(x, 0.5)
        self.assertAlmostEqual(y, 0.5, places=6)

    def test_realistic_landmark_jitter_does_not_move_the_pointer(self):
        self.pointer.update(0.40, 0.40)
        rng = np.random.default_rng(3)
        # About 1.9 px of jitter at 640 px wide: typical MediaPipe fingertip
        # noise. It must not random-walk the cursor across the screen.
        for _ in range(600):
            position = self.pointer.update(
                0.40 + float(rng.normal(0, 0.003)),
                0.40 + float(rng.normal(0, 0.003)))
        self.assertEqual(position, (0.5, 0.5))

    def test_slow_deliberate_motion_is_not_swallowed(self):
        self.pointer.update(0.40, 0.40)
        x = 0.40
        for _ in range(30):
            x += 0.003
            position = self.pointer.update(x, 0.40)
        self.assertGreater((position[0] - 0.5) * 1920, 100)

    def test_hand_reacquisition_never_jumps(self):
        self.pointer.update(0.40, 0.40)
        self.pointer.update(0.42, 0.40)
        before = self.pointer.position
        self.pointer.on_hand_lost()
        # The hand comes back somewhere completely different: the pointer
        # stays where it was and only the reference is updated.
        self.assertEqual(self.pointer.update(0.90, 0.90), before)
        self.assertEqual(self.pointer.update(0.90, 0.90), before)
        # Movement after reacquisition continues normally.
        self.assertNotEqual(self.pointer.update(0.92, 0.90), before)

    def test_fast_glitch_is_clamped(self):
        self.pointer.update(0.40, 0.40)
        x, _ = self.pointer.update(0.90, 0.40)
        # 0.5 * 1920 * 2.5 would be 2400 px; the clamp caps it at 300 px.
        self.assertAlmostEqual(x, 0.5 + 300 / 1920, places=6)

    def test_reset_can_move_to_the_real_cursor(self):
        self.pointer.update(0.40, 0.40)
        self.pointer.reset((0.8, 0.2))
        self.assertEqual(self.pointer.position, (0.8, 0.2))
        # The first sample after a reset only re-anchors.
        self.assertEqual(self.pointer.update(0.1, 0.1), (0.8, 0.2))


class HeadTrackerTests(unittest.TestCase):
    @staticmethod
    def activate(tracker, feature, anchor):
        """Feed a stable neutral window and return the resulting position."""
        position = None
        for _ in range(HEAD_NEUTRAL_SAMPLES):
            position = tracker.update(feature, anchor=anchor)
        return position

    def test_still_head_does_not_drift(self):
        tracker = HeadTracker()
        anchor = (0.62, 0.41)
        feature = (0.51, 0.49)
        self.assertEqual(self.activate(tracker, feature, anchor), anchor)
        for _ in range(240):
            position = tracker.update(feature)
        self.assertAlmostEqual(position[0], anchor[0], places=6)
        self.assertAlmostEqual(position[1], anchor[1], places=6)

    def test_alternating_jitter_stays_within_a_pixel_or_two(self):
        tracker = HeadTracker()
        feature = (0.51, 0.49)
        self.activate(tracker, feature, (0.5, 0.5))
        for index in range(240):
            wobble = 0.0005 if index % 2 else -0.0005
            position = tracker.update((feature[0] + wobble, feature[1]))
        # 0.0005 of the frame at 1080 px with a 1.8x gain is under a pixel.
        self.assertLess(abs(position[0] - 0.5) * 1920, 2.0)

    def test_head_moves_and_then_settles(self):
        tracker = HeadTracker()
        feature = (0.5, 0.5)
        self.activate(tracker, feature, (0.5, 0.5))
        moved = None
        for _ in range(60):
            moved = tracker.update((0.53, 0.5))
        self.assertGreater(moved[0], 0.52)
        settled = moved
        for _ in range(120):
            settled = tracker.update((0.53, 0.5))
        self.assertAlmostEqual(settled[0], moved[0], places=6)

    def test_first_bad_frame_does_not_become_neutral(self):
        tracker = HeadTracker()
        anchor = (0.5, 0.5)
        # The face is briefly detected in the wrong place while the user is
        # still sitting down. The stable window must discard it.
        self.assertIsNone(tracker.update((0.75, 0.75), anchor=anchor))
        position = None
        for _ in range(HEAD_NEUTRAL_SAMPLES * 2):
            position = tracker.update((0.50, 0.50), anchor=anchor)
        self.assertEqual(position, anchor)
        # A real 0.03 head move maps from the stable pose, not the outlier.
        moved = None
        for _ in range(30):
            moved = tracker.update((0.53, 0.50))
        self.assertGreater(moved[0], 0.53)
        self.assertLess(moved[0], 0.57)

    def test_pointer_is_held_while_the_neutral_window_is_moving(self):
        tracker = HeadTracker()
        anchor = (0.4, 0.6)
        for index in range(HEAD_NEUTRAL_SAMPLES * 3):
            position = tracker.update((0.30 + index * 0.01, 0.50),
                                      anchor=anchor)
            self.assertIsNone(position)
        # Once the head stops, the window can settle and the pointer starts
        # from the held position.
        for _ in range(HEAD_NEUTRAL_SAMPLES * 2):
            position = tracker.update((0.60, 0.50), anchor=anchor)
        self.assertEqual(position, anchor)

    def test_recenter_relearns_without_jumping(self):
        tracker = HeadTracker()
        self.activate(tracker, (0.50, 0.50), (0.5, 0.5))
        tracker.update((0.55, 0.50))
        tracker.reset()
        position = self.activate(tracker, (0.55, 0.50), (0.8, 0.2))
        self.assertEqual(position, (0.8, 0.2))
        self.assertAlmostEqual(tracker.update((0.55, 0.50))[0], 0.8, places=6)


class GazeMappingTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        features, targets = [], []
        for i in range(3):
            for j in range(3):
                fx = 0.18 + 0.08 * i
                fy = 0.22 + 0.07 * j
                row = gaze_vector(fx, fy, GAZE_MODEL_QUADRATIC)
                target = np.clip(row @ CURVED_RESPONSE, 0.0, 1.0)
                features.append((fx, fy))
                targets.append((float(target[0]), float(target[1])))
        self.features = features
        self.targets = targets

    def test_quadratic_fit_recovers_a_curved_eye(self):
        matrix, error = fit_gaze_mapping(self.features, self.targets)
        self.assertEqual(matrix.shape, (6, 2))
        self.assertLess(error, 1e-6)
        for (fx, fy), target in zip(self.features, self.targets):
            self.assertTrue(np.allclose(
                apply_gaze_mapping(matrix, fx, fy), target, atol=1e-6))

    def test_random_calibration_is_rejected_by_held_out_error(self):
        features = [(float(self.rng.uniform(0, 1)),
                     float(self.rng.uniform(0, 1))) for _ in range(9)]
        _, error = fit_gaze_mapping(features, TARGETS)
        self.assertGreater(error, 0.2)

    def test_legacy_affine_mapping_still_applies(self):
        matrix = np.array([[0.5, 0.1], [0.2, 0.4], [0.1, 0.1]])
        mapped = apply_gaze_mapping(matrix, 0.3, 0.4)
        expected = np.clip(np.array([1.0, 0.3, 0.4]) @ matrix, 0.0, 1.0)
        self.assertTrue(np.allclose(mapped, expected))

    def test_both_matrix_shapes_are_accepted(self):
        self.assertEqual(valid_gaze_matrix([[1, 0], [0, 1], [0, 0]]).shape, (3, 2))
        self.assertEqual(valid_gaze_matrix(
            [[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]).shape, (6, 2))
        self.assertIsNone(valid_gaze_matrix([[1, 0], [0, 1]]))
        self.assertIsNone(valid_gaze_matrix([[np.nan, 0], [0, 1], [0, 0]]))


if __name__ == "__main__":
    unittest.main()
