import ctypes
import logging
import queue
import subprocess
import threading

import pyautogui
from pywinauto import Desktop

# Disable PyAutoGUI's corner failsafe exception so edge coordinates don't crash
pyautogui.FAILSAFE = False

logger = logging.getLogger(__name__)

_user32 = ctypes.windll.user32

# Speech dictation arrives in one long phrase. PyAutoGUI types at a fixed
# interval, so this bounds how long the action thread can be busy; the cap is
# logged when it clips so a long phrase is never silently dropped.
MAX_TYPE_CHARS = 4000
TYPE_INTERVAL = 0.002
# Typing is written in small chunks so pause can stop a phrase that is already
# being typed; 32 characters at the configured interval is about 64 ms.
TYPE_CHUNK_CHARS = 32


class WindowsActions:
    """Windows controls behind the interaction engine."""

    SAFE_APPLICATIONS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
    }

    def __init__(self):
        self._cancel_check = None

    def set_cancel_check(self, check):
        """Install a callable that returns True when typing should stop.

        The action executor owns the flag; this only lets a long ``type_text``
        notice a pause between chunks instead of typing to the end.
        """
        self._cancel_check = check

    def _cancelled(self):
        check = self._cancel_check
        return bool(check and check())

    @staticmethod
    def attach_input_desktop():
        """Attach the calling thread to the active input desktop.

        SetThreadDesktop fails on a thread that owns windows, so this is only
        meaningful on the dedicated action thread, never on the GUI thread.
        """
        try:
            hdesk = _user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                _user32.SetThreadDesktop(hdesk)
                _user32.CloseDesktop(hdesk)
        except Exception:
            pass

    def screen_size(self):
        """Return physical pixel dimensions matching PyAutoGUI coordinates."""
        return tuple(pyautogui.size())

    def move_pointer(self, x, y):
        # SetCursorPos is a cheap call and does not reach a low-level mouse
        # hook, which is what lets GlobalPointerActivity ignore our own moves.
        pyautogui.moveTo(int(x), int(y), duration=0)

    def click(self, button="left", clicks=1):
        pyautogui.click(button=button, clicks=max(1, min(int(clicks), 2)))

    def scroll(self, amount):
        pyautogui.scroll(int(max(-10, min(10, amount))))

    def type_text(self, text):
        text = str(text)
        if len(text) > MAX_TYPE_CHARS:
            logger.warning("Clipping dictation of %d characters to %d", len(text), MAX_TYPE_CHARS)
            text = text[:MAX_TYPE_CHARS]
        for start in range(0, len(text), TYPE_CHUNK_CHARS):
            if self._cancelled():
                logger.info("Dictation stopped after %d of %d characters", start, len(text))
                return
            pyautogui.write(text[start:start + TYPE_CHUNK_CHARS], interval=TYPE_INTERVAL)

    def press_key(self, key):
        pyautogui.press(key)

    def open_application(self, name):
        executable = self.SAFE_APPLICATIONS.get(str(name).lower())
        if not executable:
            raise ValueError("Application is not in the safe launch list")
        subprocess.Popen([executable])

    def activate_window(self, title):
        Desktop(backend="uia").window(title=str(title)).set_focus()

    def click_control(self, window_title, control_title):
        window = Desktop(backend="uia").window(title=str(window_title))
        control = window.child_window(title=str(control_title))
        if not window.exists(timeout=1) or not control.exists(timeout=1):
            raise LookupError("The requested control was not found")
        control.click_input()


class ActionExecutor:
    """Serial worker for actions that can block for seconds.

    Typing a dictated phrase takes seconds. Running it on the Qt GUI thread
    froze the interface and delayed Emergency pause, so blocking actions are
    queued here instead. Pointer, click and scroll stay on the calling thread
    because they are sub-millisecond and must never queue behind typing.
    """

    def __init__(self, actions, on_error=None):
        self.actions = actions
        self.on_error = on_error
        self._queue = queue.Queue()
        self._thread = None
        self._stopped = False
        self._cancel = threading.Event()
        install = getattr(actions, "set_cancel_check", None)
        if install is not None:
            install(self._cancel.is_set)
        self._start()

    def _start(self):
        self._thread = threading.Thread(
            target=self._run, name="AccessFlowActions", daemon=True)
        self._thread.start()

    def _run(self):
        # Do this once per thread: it only succeeds on a thread without windows.
        # If it fails the thread must still run, otherwise every deferred action
        # (typing included) would silently stop working.
        try:
            attach = getattr(self.actions, "attach_input_desktop", None)
            if attach is not None:
                attach()
        except Exception:
            logger.exception("Could not attach the action thread to the input desktop")
        while True:
            item = self._queue.get()
            if item is None:
                return
            handler, payload = item
            try:
                handler(**payload)
            except Exception as exc:
                self._report(handler, exc)

    def _report(self, handler, exc):
        name = getattr(handler, "__name__", "action")
        logger.warning("Deferred action %s failed: %s", name, exc)
        if self.on_error:
            try:
                self.on_error(f"{name.replace('_', ' ')} failed: {exc}")
            except Exception:
                pass

    def _drain(self):
        """Empty the queue, returning how many actions were dropped."""
        dropped = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return dropped
            if item is None:
                self._queue.put(None)
                return dropped
            dropped += 1

    def submit(self, handler, payload):
        """Queue a blocking action. Order between queued actions is preserved."""
        if self._stopped:
            return
        # A new session's actions are not cancelled by an earlier pause.
        self._cancel.clear()
        self._queue.put((handler, payload))

    def cancel(self):
        """Abandon queued actions and stop one that is already running.

        Pausing must not let a dictated phrase keep typing into whatever window
        has focus, so the backlog is dropped and ``type_text`` notices the flag
        between chunks.
        """
        if self._stopped:
            return
        self._cancel.set()
        dropped = self._drain()
        if dropped:
            logger.info("Dropped %d queued action(s) on pause", dropped)

    def stop(self, timeout=2.0):
        """Drop pending work and stop the thread, e.g. when quitting."""
        if self._stopped:
            return
        self._stopped = True
        self._cancel.set()
        self._drain()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
