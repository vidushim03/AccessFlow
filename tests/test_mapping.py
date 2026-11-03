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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from accessflow.inputs.mapping import (
    GAZE_MODEL_AFFINE, GAZE_MODEL_QUADRATIC, HAND_MAX_DELTA_PX,
    DwellSelector, RelativePointer,
    apply_gaze_mapping, fit_best_gaze_mapping, fit_gaze_mapping,
    gaze_vector, valid_gaze_matrix,
)
from accessflow.inputs.vision import HEAD_NEUTRAL_SAMPLES, HeadTracker, VisionWorker

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
        # A landmark glitch must not move the cursor more than the 180 px cap.
        self.assertAlmostEqual(
            x, 0.5 + HAND_MAX_DELTA_PX / 1920, places=6)

    def test_reset_can_move_to_the_real_cursor(self):
        self.pointer.update(0.40, 0.40)
        self.pointer.reset((0.8, 0.2))
        self.assertEqual(self.pointer.position, (0.8, 0.2))
        # The first sample after a reset only re-anchors.
        self.assertEqual(self.pointer.update(0.1, 0.1), (0.8, 0.2))


class DwellSelectorTests(unittest.TestCase):
    def test_target_change_restarts_dwell_even_at_same_position(self):
        dwell = DwellSelector(duration_ms=1000, radius_px=30)
        self.assertEqual(dwell.update((100, 100), 0.0, "head", target="button-a"), (0.0, False))
        progress, clicked = dwell.update((100, 100), 0.2, "head", target="button-b")
        self.assertFalse(clicked)
        self.assertEqual(progress, 0.0)
        progress, clicked = dwell.update((100, 100), 0.4, "head", target="button-b")
        self.assertFalse(clicked)
        self.assertAlmostEqual(progress, 0.2)

    def test_same_target_completes_dwell(self):
        dwell = DwellSelector(duration_ms=500, radius_px=30)
        dwell.update((100, 100), 0.0, "gaze", target="menu-item")
        dwell.update((100, 100), 0.2, "gaze", target="menu-item")
        dwell.update((100, 100), 0.4, "gaze", target="menu-item")
        progress, clicked = dwell.update((100, 100), 0.5, "gaze", target="menu-item")
        self.assertEqual(progress, 1.0)
        self.assertTrue(clicked)


class HandDwellTests(unittest.TestCase):
    @staticmethod
    def landmarks(pose="open", index=(0.5, 0.3), pinch=False):
        points = [SimpleNamespace(x=0.5, y=0.6) for _ in range(21)]
        points[0] = SimpleNamespace(x=0.5, y=0.8)
        for mcp, pip, tip in ((5, 6, 8), (9, 10, 12),
                              (13, 14, 16), (17, 18, 20)):
            up = pose == "open" or (pose == "two_fingers" and tip in (8, 12))
            x = 0.4 + (mcp - 5) * 0.02
            points[mcp] = SimpleNamespace(x=x, y=0.62)
            points[pip] = SimpleNamespace(x=x, y=0.42)
            points[tip] = SimpleNamespace(x=x, y=0.22 if up else 0.65)
        points[8] = SimpleNamespace(x=index[0], y=index[1])
        points[4] = (SimpleNamespace(x=index[0], y=index[1]) if pinch else
                     SimpleNamespace(x=0.5, y=0.6))
        return points

    def make_worker(self, dwell_ms=500):
        worker = VisionWorker(1920, 1080, False, True,
                              preview="hand", dwell_ms=dwell_ms)
        events = []
        worker.event_ready.connect(events.append)
        worker.preview = None
        return worker, events

    def test_left_dwell_clicks_once_and_requires_move_away(self):
        worker, events = self.make_worker()
        points = self.landmarks("open")
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)  # pose debounce arms dwell
        worker._hand_tracking(points, 1.8, True)  # completes selected 500 ms
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, "click")
        self.assertEqual(events[0].payload["button"], "left")

        worker._hand_tracking(points, 2.2, True)
        worker._hand_tracking(points, 2.7, True)
        self.assertEqual(len(events), 1)

        curled = self.landmarks("other", index=(0.5, 0.37))
        worker._hand_tracking(curled, 2.8, True)  # release and move away
        worker._hand_tracking(curled, 3.4, True)
        self.assertFalse(worker.dwell_locked)
        self.assertEqual(len(events), 1)

    def test_click_latch_prevents_repeat_clicks_without_freezing_pointer(self):
        worker, events = self.make_worker(dwell_ms=500)
        still = self.landmarks("open")
        worker._hand_tracking(still, 1.0, True)
        worker._hand_tracking(still, 1.3, True)
        worker._hand_tracking(still, 1.81, True)
        self.assertTrue(worker.dwell_locked)
        self.assertEqual(len(events), 1)
        before = worker.hand_pointer.position

        # Movement before the re-arm distance still moves the pointer, but
        # the one-shot latch suppresses another click.
        worker._hand_tracking(
            self.landmarks("open", index=(0.525, 0.3)), 1.9, True)
        self.assertTrue(worker.dwell_locked)
        worker._hand_tracking(
            self.landmarks("open", index=(0.525, 0.3)), 1.92, True)
        self.assertGreater(worker.hand_pointer.position[0], before[0])
        self.assertEqual(len(events), 1)

        worker._hand_tracking(
            self.landmarks("open", index=(0.54, 0.3)), 2.0, True)
        self.assertFalse(worker.dwell_locked)
        self.assertEqual(len(events), 1)

    def test_small_index_jitter_does_not_restart_dwell(self):
        worker, events = self.make_worker(dwell_ms=500)
        worker._hand_tracking(self.landmarks("open"), 1.0, True)
        worker._hand_tracking(self.landmarks("open"), 1.3, True)
        self.assertTrue(worker.dwell_armed)

        # Typical small landmark wobble within the dwell radius should be
        # tolerated instead of restarting the countdown on every frame.
        for frame, offset in enumerate((0.008, -0.008, 0.010, -0.009, 0.006)):
            worker._hand_tracking(
                self.landmarks("open", index=(0.5 + offset, 0.3)),
                1.41 + frame * 0.1, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "left")

    def test_slow_finger_motion_cancels_an_armed_click(self):
        worker, events = self.make_worker(dwell_ms=500)
        worker._hand_tracking(self.landmarks("open"), 1.0, True)
        worker._hand_tracking(self.landmarks("open"), 1.3, True)
        self.assertTrue(worker.dwell_armed)

        # Drift slowly enough that an easing dwell anchor used to follow the
        # fingertip and still fire. A fixed anchor must cancel the click.
        for step in range(1, 21):
            worker._hand_tracking(
                self.landmarks("open", index=(0.5 + step * 0.005, 0.3)),
                1.3 + step * 0.07, True)
        # Continued slow drift may settle enough to arm a fresh dwell, but
        # it must not carry the original countdown through the movement.
        self.assertEqual(events, [])
        self.assertGreater(worker.hand_pointer.position[0], 0.5)

    def test_hand_pointer_keeps_moving_while_dwell_is_armed(self):
        worker, events = self.make_worker(dwell_ms=1000)
        still = self.landmarks("open")
        worker._hand_tracking(still, 1.0, True)
        worker._hand_tracking(still, 1.3, True)
        self.assertTrue(worker.dwell_armed)
        before = worker.hand_pointer.position

        # An intentional move must reach the pointer and cancel dwell instead
        # of being swallowed by the armed state.
        moved = self.landmarks("open", index=(0.53, 0.3))
        worker._hand_tracking(moved, 1.35, True)
        worker._hand_tracking(moved, 1.37, True)
        self.assertGreater(worker.hand_pointer.position[0], before[0])
        self.assertFalse(worker.dwell_armed)
        self.assertEqual(events, [])

    def test_hand_dwell_restarts_when_accessible_control_changes(self):
        worker, events = self.make_worker()

        class FakeTargetMonitor:
            current = "button-a"

            def set_point(self, point):
                pass

            def target(self):
                return self.current

        monitor = FakeTargetMonitor()
        worker.target_monitor = monitor
        points = self.landmarks("open")
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)
        monitor.current = "button-b"
        worker._hand_tracking(points, 1.6, True)
        worker._hand_tracking(points, 1.9, True)
        self.assertEqual(events, [])
        worker._hand_tracking(points, 2.66, True)
        worker._hand_tracking(points, 3.17, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "left")

    def test_transient_control_hit_test_does_not_restart_dwell(self):
        worker, events = self.make_worker(dwell_ms=500)

        class FakeTargetMonitor:
            current = "button-a"

            def set_point(self, point):
                pass

            def target(self):
                return self.current

        monitor = FakeTargetMonitor()
        worker.target_monitor = monitor
        points = self.landmarks("open")
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)  # target A settles, dwell arms
        worker._hand_tracking(points, 1.6, True)
        self.assertTrue(worker.dwell_armed)

        monitor.current = "button-b"
        worker._hand_tracking(points, 1.65, True)
        monitor.current = "button-a"
        worker._hand_tracking(points, 1.70, True)
        worker._hand_tracking(points, 2.11, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "left")

    def test_two_finger_pose_selects_right_click(self):
        worker, events = self.make_worker()
        points = self.landmarks("two_fingers")
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)
        worker._hand_tracking(points, 1.8, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, "click")
        self.assertEqual(events[0].payload["button"], "right")

    def test_right_click_pose_is_not_blocked_by_index_only_classifier(self):
        worker, events = self.make_worker()
        points = self.landmarks("two_fingers")
        # Keep wrist-relative geometry consistent with the V pose while making
        # the independent local index-angle classifier disagree (as can happen
        # with a tilted hand).
        points[8] = SimpleNamespace(x=0.65, y=0.32)
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)
        worker._hand_tracking(points, 1.8, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "right")

    def test_curled_middle_finger_cannot_turn_index_dwell_into_right_click(self):
        worker, events = self.make_worker()
        points = self.landmarks("two_fingers")
        # Wrist distance alone sees this middle fingertip as raised, but its
        # local joint direction shows a curl. It must stay a left click.
        points[12] = SimpleNamespace(x=0.73, y=0.32)
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)
        worker._hand_tracking(points, 1.8, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "left")

    def test_ambiguous_outer_fingers_do_not_turn_left_dwell_into_right(self):
        worker, events = self.make_worker()
        points = self.landmarks("two_fingers")
        # The ring and pinky are partly bent, but not clearly curled into a V
        # pose. They must not trigger right click just because extension
        # hysteresis classifies them as down.
        for mcp, pip, tip in ((13, 14, 16), (17, 18, 20)):
            x = points[pip].x
            points[tip] = SimpleNamespace(x=x + 0.14, y=0.36)
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)
        worker._hand_tracking(points, 1.8, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "left")

    def test_index_only_dwell_is_left_click(self):
        worker, events = self.make_worker()
        points = self.landmarks("two_fingers")
        points[12] = SimpleNamespace(x=points[12].x, y=0.65)
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)
        worker._hand_tracking(points, 1.8, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "left")

    def test_right_click_pose_survives_shorter_extended_finger_landmarks(self):
        worker, events = self.make_worker()
        points = self.landmarks("two_fingers")
        # A slightly foreshortened fingertip is common when the hand tilts.
        # Keep the fingertip aligned with the finger while shortening its
        # distal segment; this should still count as an extended middle finger.
        points[12] = SimpleNamespace(x=points[12].x, y=0.31)
        worker._hand_tracking(points, 1.0, True)
        worker._hand_tracking(points, 1.3, True)
        worker._hand_tracking(points, 1.8, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "right")

    def test_open_hand_after_two_finger_pose_returns_to_left(self):
        worker, events = self.make_worker()
        two_fingers = self.landmarks("two_fingers")
        open_hand = self.landmarks("open")
        worker._hand_tracking(two_fingers, 1.0, True)
        worker._hand_tracking(open_hand, 1.3, True)
        worker._hand_tracking(open_hand, 1.8, True)
        worker._hand_tracking(open_hand, 2.31, True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["button"], "left")

    def test_curled_index_does_not_start_left_click_dwell(self):
        worker, events = self.make_worker()
        curled = self.landmarks("other")
        curled[8] = SimpleNamespace(x=0.5, y=0.65)
        curled[4] = SimpleNamespace(x=0.5, y=0.6)
        worker._hand_tracking(curled, 1.0, True)
        worker._hand_tracking(curled, 1.3, True)
        worker._hand_tracking(curled, 2.0, True)
        self.assertEqual(events, [])

    def test_pinch_pose_does_not_trigger_a_left_click(self):
        worker, events = self.make_worker()
        open_hand = self.landmarks("open")
        worker._hand_tracking(open_hand, 1.0, True)
        worker._hand_tracking(open_hand, 1.3, True)
        self.assertTrue(worker.dwell_armed)

        pinched = self.landmarks("open", pinch=True)
        worker._hand_tracking(pinched, 1.4, True)
        worker._hand_tracking(pinched, 2.0, True)
        self.assertTrue(worker.hand_pinched)
        self.assertEqual(events, [])
        self.assertFalse(worker.dwell_armed)

        worker._hand_tracking(open_hand, 2.1, True)
        self.assertFalse(worker.hand_pinched)


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

    def test_non_finite_calibration_samples_are_rejected(self):
        features = list(self.features)
        features[3] = (float("nan"), features[3][1])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fit_gaze_mapping(features, self.targets)

    def test_quadratic_fit_recovers_a_curved_eye(self):
        matrix, error = fit_gaze_mapping(self.features, self.targets)
        self.assertEqual(matrix.shape, (6, 2))
        self.assertLess(error, 1e-6)
        for (fx, fy), target in zip(self.features, self.targets):
            self.assertTrue(np.allclose(
                apply_gaze_mapping(matrix, fx, fy), target, atol=1e-6))

    def test_model_selection_prefers_affine_for_negligible_quadratic_gain(self):
        matrix, error, model, errors = fit_best_gaze_mapping(
            self.features, self.targets)
        self.assertEqual(model, GAZE_MODEL_AFFINE)
        self.assertEqual(matrix.shape, (3, 2))
        self.assertLessEqual(
            error, errors[GAZE_MODEL_QUADRATIC] + 0.005)
        self.assertIn(GAZE_MODEL_QUADRATIC, errors)

    def test_model_selection_uses_quadratic_for_meaningful_curvature(self):
        features = [(x, y) for y in (0.2, 0.5, 0.8)
                    for x in (0.2, 0.5, 0.8)]
        targets = [(0.15 + 0.55 * x + 0.4 * x * x,
                    0.15 + 0.55 * y + 0.4 * y * y)
                   for x, y in features]
        matrix, error, model, errors = fit_best_gaze_mapping(features, targets)
        self.assertEqual(model, GAZE_MODEL_QUADRATIC)
        self.assertEqual(matrix.shape, (6, 2))
        self.assertLess(error, errors[GAZE_MODEL_AFFINE])

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
