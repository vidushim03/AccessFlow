"""Regression tests for the safety gate: pause, confidence, and failures.

Run from the project root with::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

These cover the behaviour that must not regress without a person watching:
queued work is abandoned on pause, a long dictation stops mid-phrase, low
confidence never reaches the desktop, and failed actions are reported.
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import accessflow.system.actions as actions_module
from accessflow.core.engine import InteractionEngine
from accessflow.core.events import InteractionEvent
from accessflow.system.actions import ActionExecutor, WindowsActions


class FakeActions:
    """Records every desktop action instead of touching the real desktop."""

    def __init__(self):
        self.calls = []
        self.cancel_check = None

    def set_cancel_check(self, check):
        self.cancel_check = check

    @staticmethod
    def attach_input_desktop():
        pass

    def move_pointer(self, x, y):
        self.calls.append(("move", x, y))

    def click(self, button="left", clicks=1):
        self.calls.append(("click", button))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def type_text(self, text):
        self.calls.append(("type", text))

    def press_key(self, key):
        self.calls.append(("key", key))

    def open_application(self, name):
        self.calls.append(("open", name))

    def activate_window(self, title):
        self.calls.append(("activate", title))

    def click_control(self, window_title, control_title):
        self.calls.append(("control", window_title))


class BrokenActions(FakeActions):
    def click(self, button="left", clicks=1):
        raise LookupError("no such control")


class RecordingExecutor:
    def __init__(self):
        self.submitted = []
        self.cancelled = 0

    def submit(self, handler, payload):
        self.submitted.append(handler.__name__)

    def cancel(self):
        self.cancelled += 1


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.actions = FakeActions()
        self.executor = RecordingExecutor()
        self.errors = []
        self.engine = InteractionEngine(
            self.actions, on_error=self.errors.append, executor=self.executor)

    def dispatch(self, source, action, confidence, **payload):
        return self.engine.dispatch(
            InteractionEvent(source, "test", action, confidence, payload))

    def test_pointer_moves_immediately_not_deferred(self):
        self.engine.start(["head"])
        self.dispatch("head", "move_pointer", 0.75, x=1, y=2)
        self.assertEqual(self.actions.calls, [("move", 1, 2)])
        self.assertEqual(self.executor.submitted, [])

    def test_typing_is_deferred_off_the_caller_thread(self):
        self.engine.start(["voice"])
        self.dispatch("voice", "type_text", 0.65, text="hi")
        self.assertEqual(self.executor.submitted, ["type_text"])

    def test_low_confidence_dictation_is_rejected(self):
        self.engine.start(["voice"])
        self.dispatch("voice", "type_text", 0.55, text="low")
        self.assertEqual(self.executor.submitted, [])

    def test_low_confidence_command_is_rejected(self):
        self.engine.start(["voice"])
        self.dispatch("voice", "click", 0.65, button="left")
        self.assertEqual(self.actions.calls, [])

    def test_valid_command_reaches_the_desktop(self):
        self.engine.start(["voice"])
        self.dispatch("voice", "click", 0.90, button="left")
        self.assertEqual(self.actions.calls, [("click", "left")])

    def test_disabled_source_is_ignored(self):
        self.engine.start(["head"])
        self.dispatch("voice", "click", 0.99, button="left")
        self.assertEqual(self.actions.calls, [])

    def test_paused_engine_ignores_events(self):
        self.engine.start(["head"])
        self.engine.pause()
        self.dispatch("head", "move_pointer", 0.99, x=5, y=5)
        self.assertEqual(self.actions.calls, [])

    def test_pause_cancels_queued_actions(self):
        self.engine.start(["voice"])
        self.engine.pause()
        self.assertEqual(self.executor.cancelled, 1)

    def test_pause_event_is_never_gated(self):
        # "pause" must work even from a source that is not enabled.
        self.engine.start([])
        self.assertTrue(self.dispatch("voice", "pause", 1.0))
        self.assertTrue(self.engine.paused)

    def test_failed_action_is_reported(self):
        engine = InteractionEngine(BrokenActions(), on_error=self.errors.append)
        engine.start(["voice"])
        engine.dispatch(InteractionEvent("voice", "command", "click", 0.9,
                                         {"button": "left"}))
        self.assertEqual(len(self.errors), 1)
        self.assertIn("no such control", self.errors[0])

    def test_out_of_range_confidence_is_rejected(self):
        self.engine.start(["voice"])
        self.dispatch("voice", "click", 1.5, button="left")
        self.assertEqual(self.actions.calls, [])


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.actions = FakeActions()
        self.executor = ActionExecutor(self.actions)

    def tearDown(self):
        self.executor.stop()

    def test_cancel_drops_the_backlog(self):
        started = threading.Event()
        release = threading.Event()
        ran = []

        def blocking(text):
            started.set()
            release.wait(2.0)
            ran.append(text)

        def queued(text):
            ran.append(text)

        self.executor.submit(blocking, {"text": "first"})
        self.assertTrue(started.wait(2.0), "first action never started")
        for index in range(3):
            self.executor.submit(queued, {"text": f"later{index}"})

        self.executor.cancel()
        release.set()
        # Give the worker time to finish the running action and see the drain.
        self.executor.stop()

        self.assertEqual(ran, ["first"])

    def test_cancel_does_not_block_new_session_work(self):
        self.executor.cancel()
        done = threading.Event()

        def work(text):
            done.set()

        self.executor.submit(work, {"text": "x"})
        self.assertTrue(done.wait(2.0))

    def test_stop_drains_and_joins(self):
        started = threading.Event()
        ran = []

        def slow(text):
            started.set()
            ran.append(text)

        self.executor.submit(slow, {"text": "a"})
        self.assertTrue(started.wait(2.0))
        self.executor.stop()
        self.assertFalse(self.executor._thread.is_alive())
        self.assertEqual(ran, ["a"])

    def test_submit_after_stop_is_ignored(self):
        self.executor.stop()
        self.executor.submit(lambda text: None, {"text": "x"})
        self.assertFalse(self.executor._thread.is_alive())


class TypeTextCancellationTests(unittest.TestCase):
    """A long dictation must stop being typed once the session is paused."""

    def setUp(self):
        self.actions = WindowsActions()
        self.typed = []
        write = lambda text, interval=0: self.typed.append(text)
        self.patch = mock.patch.object(actions_module.pyautogui, "write", write)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_phrase_is_not_typed_to_the_end_after_cancel(self):
        checks = {"n": 0}

        def check():
            checks["n"] += 1
            return checks["n"] > 2

        self.actions.set_cancel_check(check)
        self.actions.type_text("a" * 500)
        total = sum(len(chunk) for chunk in self.typed)
        self.assertGreater(total, 0)
        self.assertLess(total, 500)

    def test_uncancelled_phrase_is_typed_in_full(self):
        self.actions.set_cancel_check(lambda: False)
        self.actions.type_text("hello world")
        self.assertEqual("".join(self.typed), "hello world")

    def test_long_phrase_is_clipped_to_the_cap(self):
        self.actions.set_cancel_check(lambda: False)
        self.actions.type_text("b" * (actions_module.MAX_TYPE_CHARS + 500))
        self.assertEqual(sum(len(c) for c in self.typed),
                         actions_module.MAX_TYPE_CHARS)


@unittest.skipUnless(sys.platform == "win32", "Windows hooks only")
class HookFailureTests(unittest.TestCase):
    def test_failed_install_is_reported_not_swallowed(self):
        import accessflow.system.hotkey as hotkey

        messages = []
        listener = hotkey.GlobalEscapeListener()
        listener.failed.connect(messages.append)
        with mock.patch.object(hotkey._user32, "SetWindowsHookExW", return_value=0):
            listener.run()

        self.assertEqual(len(messages), 1)
        self.assertIn("Escape", messages[0])
        self.assertIn("could not be installed", messages[0])


if __name__ == "__main__":
    unittest.main()
