"""System-wide input observers used for safety and manual takeover.

Two low-level Windows hooks run on their own threads:

* ``GlobalEscapeListener`` reports Escape presses from any window so a session
  can be paused even when another application has focus.
* ``GlobalPointerActivity`` reports genuine touchpad/mouse movement so the
  camera pointer can step aside while the user drives the pointer by hand.

Neither hook consumes input: both always return the event to the chain, so the
focused application behaves normally. Only the Escape key code and the injected
flag of pointer events are inspected; positions, buttons, and timing are not
recorded or transmitted.

Keyboard hooks cannot observe input aimed at elevated (administrator) windows.
"""

from __future__ import annotations

import ctypes
import sys
import time

from PySide6.QtCore import QThread, Signal

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14

WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_QUIT = 0x0012
VK_ESCAPE = 0x1B
PM_NOREMOVE = 0x0000

WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_RBUTTONDOWN = 0x0204
WM_MBUTTONDOWN = 0x0207
WM_MOUSEWHEEL = 0x020A
WM_XBUTTONDOWN = 0x020B
WM_MOUSEHWHEEL = 0x020E
LLMHF_INJECTED = 0x00000001

_POINTER_MESSAGES = frozenset({
    WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_MBUTTONDOWN,
    WM_MOUSEWHEEL, WM_XBUTTONDOWN, WM_MOUSEHWHEEL,
})

if sys.platform == "win32":
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t)

    _user32.SetWindowsHookExW.restype = ctypes.c_void_p
    _user32.SetWindowsHookExW.argtypes = (ctypes.c_int, _HOOKPROC, ctypes.c_void_p, wintypes.DWORD)
    _user32.CallNextHookEx.restype = ctypes.c_ssize_t
    _user32.CallNextHookEx.argtypes = (
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t)
    _user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    _user32.UnhookWindowsHookEx.argtypes = (ctypes.c_void_p,)
    _user32.GetMessageW.restype = wintypes.BOOL
    _user32.GetMessageW.argtypes = (
        ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
    _user32.PeekMessageW.restype = wintypes.BOOL
    _user32.PeekMessageW.argtypes = (
        ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT)
    _user32.TranslateMessage.argtypes = (ctypes.POINTER(wintypes.MSG),)
    _user32.DispatchMessageW.argtypes = (ctypes.POINTER(wintypes.MSG),)
    _user32.PostThreadMessageW.restype = wintypes.BOOL
    _user32.PostThreadMessageW.argtypes = (
        wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    _kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    class _KeyboardHookStruct(ctypes.Structure):
        _fields_ = [
            ("vkCode", wintypes.DWORD),
            ("scanCode", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _MouseHookStruct(ctypes.Structure):
        _fields_ = [
            ("pt", wintypes.POINT),
            ("mouseData", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]


class _GlobalHook(QThread):
    """Run a low-level Windows hook on its own message-pumping thread."""

    #: Human-readable name used when the hook cannot be installed.
    LABEL = "Global hook"
    #: What the user can still rely on if installation fails.
    FALLBACK = "the in-app controls still work"

    failed = Signal(str)

    def __init__(self, hook_id, parent=None):
        super().__init__(parent)
        self._hook_id = hook_id
        self._thread_id = 0
        self._hook = None
        self._stopping = False
        self._callback = _HOOKPROC(self._dispatch) if sys.platform == "win32" else None

    def _dispatch(self, n_code, w_param, l_param):
        try:
            if n_code >= 0 and not self._stopping:
                self.handle(w_param, l_param)
        except Exception:
            pass
        return _user32.CallNextHookEx(None, n_code, w_param, l_param)

    def handle(self, w_param, l_param):
        raise NotImplementedError

    def run(self):
        if sys.platform != "win32" or self._stopping:
            return
        self._thread_id = _kernel32.GetCurrentThreadId()
        message = wintypes.MSG()
        # Force this thread's message queue into existence before stop() can post.
        _user32.PeekMessageW(ctypes.byref(message), None, 0, 0, PM_NOREMOVE)
        self._hook = _user32.SetWindowsHookExW(self._hook_id, self._callback, None, 0)
        if not self._hook or self._stopping:
            if self._hook:
                _user32.UnhookWindowsHookEx(self._hook)
                self._hook = None
            self._thread_id = 0
            if not self._stopping:
                # Never let the app keep promising a shortcut it does not have.
                self.failed.emit(
                    f"{self.LABEL} could not be installed; {self.FALLBACK}")
            return
        try:
            while _user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                _user32.TranslateMessage(ctypes.byref(message))
                _user32.DispatchMessageW(ctypes.byref(message))
        finally:
            _user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
            self._thread_id = 0

    def stop(self, timeout=500):
        self._stopping = True
        if sys.platform == "win32" and self._thread_id:
            _user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self.isRunning():
            self.wait(timeout)


class GlobalEscapeListener(_GlobalHook):
    """Emit ``escaped`` when Escape is pressed on any window."""

    LABEL = "The system-wide Escape shortcut"
    FALLBACK = "Escape still pauses while an AccessFlow window has focus"

    escaped = Signal()

    def __init__(self, parent=None):
        super().__init__(WH_KEYBOARD_LL, parent)

    def handle(self, w_param, l_param):
        if w_param not in (WM_KEYDOWN, WM_SYSKEYDOWN):
            return
        info = ctypes.cast(l_param, ctypes.POINTER(_KeyboardHookStruct)).contents
        if info.vkCode == VK_ESCAPE:
            self.escaped.emit()


class GlobalPointerActivity(_GlobalHook):
    """Emit ``activity`` for real touchpad/mouse input, never for our own moves.

    AccessFlow moves the pointer with SetCursorPos, which does not reach a
    low-level mouse hook, and PyAutoGUI's clicks are flagged as injected.
    """

    activity = Signal()

    LABEL = "Touchpad takeover detection"
    FALLBACK = "camera pointer output continues until you pause"

    def __init__(self, parent=None):
        super().__init__(WH_MOUSE_LL, parent)
        self._last_emit = 0.0

    def handle(self, w_param, l_param):
        if w_param not in _POINTER_MESSAGES:
            return
        info = ctypes.cast(l_param, ctypes.POINTER(_MouseHookStruct)).contents
        if info.flags & LLMHF_INJECTED:
            return
        now = time.monotonic()
        # A fast mouse can report hundreds of events per second; one signal is
        # enough to extend the takeover window.
        if now - self._last_emit < 0.15:
            return
        self._last_emit = now
        self.activity.emit()
