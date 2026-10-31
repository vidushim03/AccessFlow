import ctypes
import subprocess
import pyautogui
from pywinauto import Desktop

# Disable PyAutoGUI's corner failsafe exception so edge coordinates don't crash
pyautogui.FAILSAFE = False

_user32 = ctypes.windll.user32


class WindowsActions:
    """Windows controls behind the interaction engine."""

    SAFE_APPLICATIONS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
    }

    def __init__(self):
        # Ensure process thread attaches to active input desktop
        self._attach_input_desktop()

    @staticmethod
    def _attach_input_desktop():
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
        self._attach_input_desktop()
        pyautogui.moveTo(int(x), int(y), duration=0)

    def click(self, button="left", clicks=1):
        pyautogui.click(button=button, clicks=max(1, min(int(clicks), 2)))

    def scroll(self, amount):
        pyautogui.scroll(int(max(-10, min(10, amount))))

    def type_text(self, text):
        pyautogui.write(str(text)[:1000], interval=0.005)

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

