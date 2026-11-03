"""Camera backend selection tests that do not require a camera device."""
from __future__ import annotations

import unittest
from unittest import mock

import cv2

from accessflow.inputs.vision import CAMERA_BACKENDS, CameraUnavailableError, open_camera


class FakeCapture:
    def __init__(self, opened, readable=False):
        self.opened = opened
        self.readable = readable
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def read(self):
        return self.readable, object() if self.readable else None

    def release(self):
        self.released = True


class CameraOpenTests(unittest.TestCase):
    def test_tries_another_backend_when_first_open_does_not_deliver_frames(self):
        closed = FakeCapture(False)
        no_frame = FakeCapture(True, readable=False)
        working = FakeCapture(True, readable=True)
        with mock.patch("accessflow.inputs.vision.cv2.VideoCapture",
                        side_effect=[closed, no_frame, working]) as video_capture:
            capture, label = open_camera()

        self.assertIs(capture, working)
        self.assertIn(CAMERA_BACKENDS[2][0], label)
        self.assertEqual(video_capture.call_args_list[0].args, (0, cv2.CAP_DSHOW))
        self.assertEqual(video_capture.call_args_list[1].args, (0, cv2.CAP_MSMF))
        self.assertEqual(video_capture.call_args_list[2].args, (0, cv2.CAP_ANY))
        self.assertTrue(closed.released)
        self.assertTrue(no_frame.released)
        self.assertFalse(working.released)
        self.assertEqual(len(working.set_calls), 2)

    def test_tries_second_camera_device_after_first_device_fails(self):
        failures = [FakeCapture(False) for _ in CAMERA_BACKENDS]
        working = FakeCapture(True, readable=True)
        with mock.patch("accessflow.inputs.vision.cv2.VideoCapture",
                        side_effect=[*failures, working]) as video_capture:
            capture, label = open_camera()

        self.assertIs(capture, working)
        self.assertEqual(video_capture.call_args_list[-1].args,
                         (1, CAMERA_BACKENDS[0][1]))
        self.assertIn("Camera 1", label)
        self.assertTrue(all(item.released for item in failures))

    def test_reports_recovery_steps_when_no_device_works(self):
        failures = [FakeCapture(False) for _ in range(2 * len(CAMERA_BACKENDS))]
        with mock.patch("accessflow.inputs.vision.cv2.VideoCapture",
                        side_effect=failures):
            with self.assertRaisesRegex(
                    CameraUnavailableError,
                    "Close another app using the camera.*camera permissions"):
                open_camera()
        self.assertTrue(all(item.released for item in failures))


if __name__ == "__main__":
    unittest.main()
