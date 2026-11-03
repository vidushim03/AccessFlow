import ctypes
import logging
import queue
import re
import subprocess
import threading

logger = logging.getLogger(__name__)

_user32 = ctypes.windll.user32


def _pyautogui():
    """Import/configure PyAutoGUI only after Qt has initialized Windows DPI.

    PyAutoGUI sets process DPI awareness during import. Importing it while
    loading this module used to run before QApplication and could lock the
    process to a different DPI mode from Qt's per-monitor configuration.
    """
    import pyautogui

    pyautogui.FAILSAFE = False
    # Camera tracking emits frequent cursor updates; no artificial pause is
    # needed between instantaneous move/click calls.
    pyautogui.PAUSE = 0
    return pyautogui

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
    def inspect_target(x, y):
        """Return a stable UIA identity and label for the control at a point.

        The identity lets dwell reset when the pointer moves from one exposed
        control to another, even if both controls are close together. Some
        applications do not expose UIA elements; those use spatial dwell instead.
        """
        x, y = int(x), int(y)
        try:
            from pywinauto import Desktop

            element = Desktop(backend="uia").from_point(x, y)
            for _ in range(8):
                info = element.element_info
                control_type = info.control_type or ""
                if (control_type in {
                        "Button", "CheckBox", "ComboBox", "Document", "Edit",
                        "Hyperlink", "ListItem",
                        "MenuItem", "RadioButton", "SplitButton", "TabItem", "TreeItem",
                    } and element.is_visible() and element.is_enabled()):
                    rect = info.rectangle
                    runtime_id = getattr(info, "runtime_id", None)
                    identity = ("uia", getattr(info, "process_id", None),
                                getattr(info, "handle", None),
                                runtime_id or control_type, info.name or "", control_type,
                                rect.left, rect.top, rect.right, rect.bottom)
                    return identity, (info.name or control_type)
                element = element.parent()
                if element is None:
                    break
        except Exception as exc:
            logger.debug("UIA target lookup unavailable at (%d,%d): %s", x, y, exc)
        # A coordinate fallback cannot identify a semantic control. Let dwell
        # use its normal spatial tolerance instead of resetting on every pixel.
        return None, "screen position"

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
        return tuple(_pyautogui().size())

    def move_pointer(self, x, y):
        # SetCursorPos is a cheap call and does not reach a low-level mouse
        # hook, which is what lets GlobalPointerActivity ignore our own moves.
        _pyautogui().moveTo(int(x), int(y), duration=0)

    def click(self, button="left", clicks=1):
        _pyautogui().click(button=button, clicks=max(1, min(int(clicks), 2)))

    def click_target(self, button="left", x=None, y=None):
        """Activate the UI Automation control at the click point if possible.

        Dwell-based pointer selection benefits from invoking the exposed screen
        control (for example, a Start-menu item) rather than sending a blind
        coordinate click. Many custom surfaces expose no UIA control; for those
        cases retain the normal mouse click as a fallback.
        """
        pyautogui = _pyautogui()
        if x is None or y is None:
            x, y = pyautogui.position()
        x, y = int(x), int(y)
        if button == "left":
            try:
                from pywinauto import Desktop

                element = Desktop(backend="uia").from_point(x, y)
                actionable = {
                    "Button", "CheckBox", "ComboBox", "Document", "Edit",
                    "Hyperlink", "ListItem",
                    "MenuItem", "RadioButton", "SplitButton", "TabItem", "TreeItem",
                }
                for _ in range(8):
                    info = element.element_info
                    if (info.control_type in actionable and element.is_visible()
                            and element.is_enabled()):
                        try:
                            element.invoke()
                            return
                        except Exception:
                            try:
                                element.click_input()
                                return
                            except Exception:
                                break
                    element = element.parent()
                    if element is None:
                        break
            except Exception as exc:
                logger.debug("UIA target click fell back to mouse input: %s", exc)
        pyautogui.click(x=x, y=y, button=button)

    def scroll(self, amount):
        _pyautogui().scroll(int(max(-10, min(10, amount))))

    def type_text(self, text):
        text = str(text)
        if len(text) > MAX_TYPE_CHARS:
            logger.warning("Clipping dictation of %d characters to %d", len(text), MAX_TYPE_CHARS)
            text = text[:MAX_TYPE_CHARS]
        for start in range(0, len(text), TYPE_CHUNK_CHARS):
            if self._cancelled():
                logger.info("Dictation stopped after %d of %d characters", start, len(text))
                return
            _pyautogui().write(text[start:start + TYPE_CHUNK_CHARS], interval=TYPE_INTERVAL)

    def press_key(self, key):
        _pyautogui().press(key)

    def open_application(self, name):
        name = str(name).lower()
        executable = self.SAFE_APPLICATIONS.get(name)
        if not executable:
            raise ValueError("Application is not in the safe launch list")
        subprocess.Popen([executable])
        # The voice worker queues dictation behind this action. Wait for the
        # launched window and focus it so “open Notepad” followed by “type …”
        # cannot race and type into the previously focused application.
        from pywinauto import Desktop

        title_pattern = rf"(?i).*{re.escape(name)}.*"
        try:
            window = Desktop(backend="uia").window(title_re=title_pattern)
            window.wait("visible", timeout=8)
            window.set_focus()
        except Exception as exc:
            raise RuntimeError(
                f"{name.title()} opened, but its window could not be focused. "
                "Select the text area before dictating.") from exc

    def activate_window(self, title):
        # pywinauto initializes COM when imported. UI Automation runs on the
        # dedicated action thread so it cannot set the Qt UI thread to MTA
        # before Qt's OLE/STA initialization.
        from pywinauto import Desktop

        Desktop(backend="uia").window(title=str(title)).set_focus()

    def click_control(self, window_title, control_title):
        from pywinauto import Desktop

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
        self._generation = 0
        self._current_generation = 0
        self._generation_lock = threading.Lock()
        install = getattr(actions, "set_cancel_check", None)
        if install is not None:
            install(self._is_cancelled)
        self._start()

    def _is_cancelled(self):
        with self._generation_lock:
            return (self._stopped
                    or self._current_generation != self._generation)

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
            handler, payload, generation = item
            with self._generation_lock:
                if self._stopped or generation != self._generation:
                    continue
                self._current_generation = generation
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
        with self._generation_lock:
            if self._stopped:
                return
            generation = self._generation
        self._queue.put((handler, payload, generation))

    def cancel(self):
        """Abandon queued actions and stop one that is already running.

        Pausing must not let a dictated phrase keep typing into whatever window
        has focus, so the backlog is dropped and ``type_text`` notices the flag
        between chunks.
        """
        with self._generation_lock:
            if self._stopped:
                return
            self._generation += 1
        dropped = self._drain()
        if dropped:
            logger.info("Dropped %d queued action(s) on pause", dropped)

    def stop(self, timeout=2.0):
        """Drop pending work and stop the thread, e.g. when quitting."""
        with self._generation_lock:
            if self._stopped:
                return
            self._stopped = True
            self._generation += 1
        self._drain()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
