import subprocess
import pyautogui
from pywinauto import Desktop


class WindowsActions:
    """Windows controls behind the interaction engine."""

    SAFE_APPLICATIONS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
    }

    def move_pointer(self, x, y):
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

