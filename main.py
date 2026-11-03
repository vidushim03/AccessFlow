from __future__ import annotations

import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QSettings, Signal, QObject, QTimer, QRect, QSize
from PySide6.QtGui import (
    QAction, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton, QSlider,
    QGridLayout, QScrollArea, QStackedWidget, QSystemTrayIcon, QVBoxLayout,
    QWidget,
)

from accessflow.core.engine import InteractionEngine
from accessflow.system.hotkey import GlobalEscapeListener, GlobalPointerActivity
from accessflow.system.actions import ActionExecutor, WindowsActions
from accessflow.inputs.mapping import (
    GAZE_MODEL_AFFINE, GAZE_MODEL_QUADRATIC, fit_best_gaze_mapping,
    leave_one_out_gaze_predictions, model_for_matrix, valid_gaze_matrix,
)
from accessflow.inputs.vision import (
    HAND_CONNECTIONS, PREVIEW_FACE_POINTS, VisionWorker,
)
from accessflow.inputs.voice import VoiceModelDownloadWorker, VoiceWorker

APP_NAME = "AccessFlow"

# Gaze calibration is accepted only after the eye feature has settled. The
# research Eye Control Mouse reference uses a 0.07 standard-deviation limit
# for iris position normalized by eye width. AccessFlow uses that feature too.
CALIBRATION_STABLE_STD = 0.07
CALIBRATION_MIN_SAMPLES = 15
# Frames right after a target appears are transition frames: the eye needs time
# to saccade to the new dot and settle, and on the first dot the user needs
# time to locate it. 1.5 s keeps transition/saccade samples out of the stability
# window without making each point feel sluggish.
CALIBRATION_SAMPLE_GRACE = 1.5
CALIBRATION_MIN_DWELL = 2.5
CALIBRATION_POINT_TIMEOUT = 12.0
# Leave-one-out position error in normalized screen coordinates. Keep the
# limit conservative so a clearly unstable mapping is never activated.
CALIBRATION_MAX_ERROR = 0.08
GAZE_FEATURE = "iris_in_eye_v1"
CALIBRATION_KEY = "gaze/calibration"


def _valid_gaze_matrix(value):
    """Return a usable affine or quadratic mapping as a list, or None."""
    array = valid_gaze_matrix(value)
    return None if array is None else array.tolist()


# The page order is shared by the sidebar buttons and programmatic navigation.
PAGE_DASHBOARD = 0
PAGE_INPUTS = 1
PAGE_PREVIEW = 2
PAGE_ACCESSIBILITY = 3
PAGE_ABOUT = 4


class CalibrationOverlay(QWidget):
    cancelled = Signal()
    def __init__(self, target_index, parent=None):
        super().__init__(parent)
        self.target_index = target_index
        self.targets = [(0.12, 0.12), (0.5, 0.12), (0.88, 0.12), (0.12, 0.5),
                        (0.5, 0.5), (0.88, 0.5), (0.12, 0.88), (0.5, 0.88), (0.88, 0.88)]
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.showFullScreen()
        self.activateWindow()
        self.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 25, 40, 242))
        painter.setPen(QColor("white"))
        painter.drawText(QRect(0, 32, self.width(), 55), Qt.AlignmentFlag.AlignCenter,
                         "Look at the red dot and keep your head still · Esc cancels")
        x, y = self.targets[self.target_index]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ff5252"))
        painter.drawEllipse(int(x * self.width()) - 14, int(y * self.height()) - 14, 28, 28)

class TrackingPreviewWidget(QWidget):
    """Live camera + landmark + mapped-cursor display for one input method.

    The left side is the mirrored camera feed with the detected landmarks.
    The right side is a miniature screen showing exactly where the mapping
    would put the pointer, so the user can see the mapping instead of
    guessing from a cursor that jumps around on the real desktop.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload = None
        self.setMinimumSize(560, 360)
        self.setObjectName("trackingPreview")

    def sizeHint(self):
        return QSize(760, 460)

    def clear(self):
        self._payload = None
        self.update()

    def set_payload(self, payload):
        # Taking a deep copy of the frame means the camera thread can reuse
        # its buffer immediately; painting happens later on the GUI thread.
        frame = payload.get("frame")
        image = None
        if frame is not None and getattr(frame, "ndim", 0) == 3:
            height, width = frame.shape[:2]
            image = QImage(frame.data, width, height, frame.strides[0],
                           QImage.Format.Format_RGB888).copy()
        payload = dict(payload)
        payload["image"] = image
        payload.pop("frame", None)
        self._payload = payload
        self.update()

    def _video_rect(self):
        margin = 14
        available_w = self.width() - 2 * margin
        available_h = self.height() - 2 * margin
        video_w = int(available_w * 0.62)
        video_h = min(available_h, int(video_w * 3 / 4))
        return QRect(margin, margin, video_w, video_h)

    def _screen_rect(self):
        margin = 14
        video = self._video_rect()
        left = video.right() + margin
        width = self.width() - left - margin
        height = self.height() - 2 * margin
        return QRect(left, margin, max(120, width), height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0b1624"))

        video = self._video_rect()
        screen = self._screen_rect()
        self._paint_screen(painter, screen)
        self._paint_video(painter, video)

        payload = self._payload
        if payload is None:
            painter.setPen(QColor("#8fa8bf"))
            painter.drawText(video, Qt.AlignmentFlag.AlignCenter,
                             "Choose Hand, Eye, or Head above to start the preview.")
            return
        label = payload.get("label") or ""
        if label:
            painter.setPen(QColor("#dbeafe"))
            font = QFont(self.font())
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(video.adjusted(8, video.height() - 34, -8, -6),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             label)

    @staticmethod
    def _paint_screen(painter, rect):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#101f31"))
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QPen(QColor("#27415c"), 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)
        painter.setPen(QColor("#7f9ab3"))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(rect.adjusted(12, 8, -12, 0),
                         Qt.AlignmentFlag.AlignLeft, "MAPPED POINTER")
        inner = rect.adjusted(18, 34, -18, -18)
        painter.setPen(QPen(QColor("#1c3550"), 1))
        painter.setBrush(QColor("#08111c"))
        painter.drawRoundedRect(inner, 6, 6)
        # Centre guides make it obvious when the mapping is stuck in a corner.
        painter.setPen(QPen(QColor("#18283a"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(inner.left(), inner.center().y(), inner.right(), inner.center().y())
        painter.drawLine(inner.center().x(), inner.top(), inner.center().x(), inner.bottom())

    def _paint_video(self, painter, rect):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#08111c"))
        painter.drawRoundedRect(rect, 10, 10)
        payload = self._payload
        if not payload:
            return
        image = payload.get("image")
        if image is None or image.isNull():
            return
        scaled = image.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        target = QRect(0, 0, scaled.width(), scaled.height())
        target.moveCenter(rect.center())
        painter.drawImage(target, scaled)

        points = payload.get("points") or []
        hand = payload.get("mode") == "hand" and len(points) >= 21

        def to_widget(index):
            px, py = points[index]
            return (target.left() + px * target.width(),
                    target.top() + py * target.height())

        if hand:
            painter.setPen(QPen(QColor("#38bdf8"), 2))
            for a, b in HAND_CONNECTIONS:
                x1, y1 = to_widget(a)
                x2, y2 = to_widget(b)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        for index, point in enumerate(points):
            px, py = point
            x = target.left() + px * target.width()
            y = target.top() + py * target.height()
            if hand:
                color = QColor("#f59e0b") if index == 8 else QColor("#e2f3ff")
                radius = 6 if index == 8 else 3
            else:
                color = QColor("#f472b6") if index in (468, 473) else QColor("#67e8f9")
                radius = 5 if index in (468, 473) else 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)

        cursor = payload.get("cursor")
        if cursor is None:
            return
        screen = self._screen_rect().adjusted(18, 34, -18, -18)
        cx = screen.left() + max(0.0, min(1.0, cursor[0])) * screen.width()
        cy = screen.top() + max(0.0, min(1.0, cursor[1])) * screen.height()
        painter.setPen(QPen(QColor("#ef4444"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(screen.left(), int(cy), screen.right(), int(cy))
        painter.drawLine(int(cx), screen.top(), int(cx), screen.bottom())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ef4444"))
        painter.drawEllipse(int(cx - 6), int(cy - 6), 12, 12)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(int(cx - 2), int(cy - 2), 4, 4)
        if payload.get("mode") in ("gaze", "head"):
            progress = max(0.0, min(1.0, float(payload.get("dwell_progress", 0.0))))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#93a8bb"), 2))
            painter.drawEllipse(int(cx - 14), int(cy - 14), 28, 28)
            if progress > 0:
                painter.setPen(QPen(QColor("#34d399"), 3))
                painter.drawArc(QRect(int(cx - 14), int(cy - 14), 28, 28),
                                90 * 16, -int(progress * 360 * 16))
        if not payload.get("calibrated", True):
            painter.setPen(QColor("#fbbf24"))
            painter.drawText(screen.adjusted(10, 0, -10, -8),
                             Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                             "Not calibrated")


class Bus(QObject):
    status = Signal(str)


class AccessFlowKeyboard(QWidget):
    """Non-activating on-screen keyboard for hand/head/eye pointer users."""

    key_requested = Signal(str, bool)

    def __init__(self, parent=None):
        flags = (Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.WindowDoesNotAcceptFocus)
        super().__init__(parent, flags)
        self.setWindowTitle("AccessFlow Keyboard")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumWidth(560)
        self.setWindowOpacity(0.98)
        self._shifted = False
        self._caps_locked = False

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("AccessFlow keyboard")
        title.setStyleSheet("font-size: 17px; font-weight: 700; padding: 4px;")
        header.addWidget(title)
        header.addStretch()
        close = QPushButton("Close")
        close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close.clicked.connect(self.hide)
        header.addWidget(close)
        layout.addLayout(header)
        help_text = QLabel(
            "Use your active pointer to click or dwell on the text field, then dwell "
            "on a key here. This window stays open without taking text focus.")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        grid = QGridLayout()
        grid.setSpacing(5)
        rows = [
            list("1234567890") + ["Backspace"],
            list("qwertyuiop"),
            list("asdfghjkl") + ["Enter"],
            list("zxcvbnm") + [".", ",", "Caps", "Shift"],
        ]
        for row_index, row in enumerate(rows):
            for col_index, label in enumerate(row):
                button = QPushButton(label)
                button.setMinimumSize(42 if len(label) == 1 else 76, 42)
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                if label in ("Caps", "Shift"):
                    button.setCheckable(True)
                    if label == "Caps":
                        self.caps_button = button
                    else:
                        self.shift_button = button
                button.clicked.connect(
                    lambda checked=False, key=label: self._select_key(key))
                grid.addWidget(button, row_index, col_index)

        special = [
            ("Tab", "tab"), ("Space", "space"), ("←", "left"),
            ("↑", "up"), ("↓", "down"), ("→", "right"),
        ]
        row_index = len(rows)
        for col_index, (label, key) in enumerate(special):
            button = QPushButton(label)
            button.setMinimumSize(64, 42)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(
                lambda checked=False, value=key: self.key_requested.emit(value, True))
            grid.addWidget(button, row_index, col_index)
        layout.addLayout(grid)

    def _select_key(self, key):
        if key == "Shift":
            self._shifted = not self._shifted
            self.shift_button.setChecked(self._shifted)
            return
        if key == "Caps":
            self._caps_locked = not self._caps_locked
            self.caps_button.setChecked(self._caps_locked)
            return
        if key == "Backspace":
            self.key_requested.emit("backspace", True)
            return
        if key == "Enter":
            self.key_requested.emit("enter", True)
            return
        shifted_symbols = {
            "1": "!", "2": "@", "3": "#", "4": "$", "5": "%",
            "6": "^", "7": "&", "8": "*", "9": "(", "0": ")",
            ".": ">", ",": "<",
        }
        if self._shifted and key in shifted_symbols:
            character = shifted_symbols[key]
        elif len(key) == 1 and key.isalpha() and (
                self._shifted ^ getattr(self, "_caps_locked", False)):
            character = key.upper()
        else:
            character = key
        self.key_requested.emit(character, False)
        if self._shifted:
            self._shifted = False
            self.shift_button.setChecked(False)


class DwellFeedbackOverlay(QWidget):
    """Visible progress ring that never intercepts clicks or focus."""

    def __init__(self, parent=None):
        flags = (Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                 | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.WindowDoesNotAcceptFocus)
        super().__init__(parent, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(48, 48)
        self.progress = 0.0
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)

    def show_progress(self, point, progress, scale=1.0):
        self.progress = max(0.0, min(1.0, float(progress)))
        if self.progress <= 0:
            self.hide_timer.stop()
            self.hide()
            return
        x, y = int(point[0] / scale), int(point[1] / scale)
        self.move(x - self.width() // 2, y - self.height() // 2)
        self.update()
        self.show()
        self.raise_()
        if self.progress >= 1.0:
            self.hide_timer.start(450)
        else:
            # A camera dropout or handoff must not leave a stale ring on screen.
            # Live progress updates keep renewing this short lease.
            self.hide_timer.start(700)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(QPen(QColor(15, 28, 42, 190), 5))
        painter.drawEllipse(rect)
        painter.setPen(QPen(QColor("#58d6a5"), 4, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 90 * 16, -int(self.progress * 360 * 16))


class AccessFlowWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("Vidushi", APP_NAME)
        self.bus = Bus()
        self.actions = WindowsActions()
        # Typing and UI automation run on their own thread so the interface never
        # freezes mid-dictation and Emergency pause stays responsive.
        self.action_executor = ActionExecutor(self.actions, on_error=self._report_error)
        self.engine = InteractionEngine(
            self.actions, on_pause=self._pause_workers,
            on_error=self._report_error, executor=self.action_executor)
        self.vision_worker = None
        self.voice_worker = None
        # Workers keep tearing down MediaPipe/audio after a pause request, so they
        # stay referenced here until their finished signal arrives.
        self._retiring_workers = []
        self._pending_start = False
        self._pending_calibration = False
        self._pending_preview = None
        self._pending_pointer_event = None
        self._quitting = False
        self._calibration_in_flight = False
        self.running = False
        self.calibration_notice = "Eye gaze input is disabled in this version."
        self.safety_warning = ""
        # Retain the legacy calibration helpers for compatibility, but don't
        # load or apply an old gaze mapping in the active input flow.
        self.gaze_transform = None
        # Samples for the current target, and the accepted mean per target.
        self.calibration_buffer = deque(maxlen=30)
        self.calibration_points = {}
        self.calibration_target_index = 0
        self.calibration_target_started = 0.0
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self._poll_calibration)
        self.pointer_move_timer = QTimer(self)
        self.pointer_move_timer.setSingleShot(True)
        self.pointer_move_timer.setInterval(16)
        self.pointer_move_timer.timeout.connect(self._flush_pointer_move)
        self.calibration_overlay = None
        self.download_worker = None
        self.preview_worker = None
        self.preview_mode = None
        self._pending_preview = None
        self.setWindowTitle("AccessFlow")
        self.resize(1120, 740)
        self.setMinimumSize(920, 620)
        self._build_ui()
        self.accessflow_keyboard = AccessFlowKeyboard(self)
        self.accessflow_keyboard.key_requested.connect(self._keyboard_key_requested)
        self.dwell_feedback_overlay = DwellFeedbackOverlay(self)
        self._build_tray()
        self._restore_settings()
        self.esc_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.esc_shortcut.activated.connect(self._pause)
        self.esc_listener = GlobalEscapeListener(self)
        self.esc_listener.escaped.connect(self._global_escape)
        self.esc_listener.failed.connect(self._hook_failed)
        self.pointer_listener = GlobalPointerActivity(self)
        self.pointer_listener.activity.connect(self._touchpad_takeover)
        self.pointer_listener.failed.connect(self._hook_failed)
        if sys.platform == "win32":
            self.esc_listener.start()
            self.pointer_listener.start()
        QApplication.instance().aboutToQuit.connect(self._stop_global_listeners)

    def _build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(232)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(22, 28, 18, 22)
        brand = QLabel("AccessFlow")
        brand.setObjectName("brand")
        side.addWidget(brand)
        side.addWidget(QLabel("ADAPTIVE COMPUTER ACCESS"))
        side.addSpacing(24)

        self.pages = QStackedWidget()
        self.nav_buttons = []
        for index, name in enumerate(["Dashboard", "Input methods",
                                      "Tracking preview", "Accessibility", "About"]):
            button = QPushButton(name)
            button.setObjectName("nav")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, i=index: self._navigate(i))
            side.addWidget(button)
            self.nav_buttons.append(button)
        side.addStretch()
        side.addWidget(QLabel("SESSION"))
        self.status_label = QLabel("Paused")
        self.status_label.setObjectName("sideStatus")
        side.addWidget(self.status_label)
        # Worker and action threads report through the bus so widgets are only
        # touched on the GUI thread.
        self.bus.status.connect(self.status_label.setText)

        for page in (self._dashboard(), self._inputs(), self._preview(),
                     self._accessibility(), self._about()):
            self.pages.addWidget(page)
        layout.addWidget(sidebar)
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)
        self._navigate(0)

    def _page(self, title, subtitle):
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(38, 32, 38, 30)
        box.setSpacing(18)
        heading = QLabel(title)
        heading.setObjectName("heading")
        box.addWidget(heading)
        desc = QLabel(subtitle)
        desc.setObjectName("subtitle")
        desc.setWordWrap(True)
        box.addWidget(desc)
        return page, box

    def _card(self, title, detail):
        card = QFrame()
        card.setObjectName("card")
        column = QVBoxLayout(card)
        header = QLabel(title)
        header.setObjectName("cardTitle")
        column.addWidget(header)
        body = QLabel(detail)
        body.setObjectName("cardText")
        body.setWordWrap(True)
        column.addWidget(body)
        return card

    def _dashboard(self):
        page, box = self._page("Welcome to AccessFlow", "Your assistive controls are ready when you are.")
        row = QHBoxLayout()
        self.session_card = self._card("Session paused", "Camera and microphone are off.")
        row.addWidget(self.session_card, 2)
        row.addWidget(self._card("Safety control", "Press Escape or use Emergency pause at any time."), 1)
        box.addLayout(row)

        self.start_button = QPushButton("Start selected inputs")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(48)
        self.start_button.clicked.connect(self._toggle_running)
        box.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignLeft)

        pause = QPushButton("Emergency pause")
        pause.setMinimumHeight(44)
        pause.clicked.connect(self._pause)
        box.addWidget(pause, 0, Qt.AlignmentFlag.AlignLeft)

        note = QLabel("Choose hand, head, or voice input. Camera tracking uses your webcam; voice input needs an offline speech model.")
        note.setObjectName("notice")
        note.setWordWrap(True)
        self.dashboard_note = note
        box.addWidget(note)
        box.addStretch()
        return page

    def _inputs(self):
        page, box = self._page("Input methods", "Select what AccessFlow should listen to when you start a session.")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 12, 8)
        content_layout.setSpacing(14)
        scroll.setWidget(content)
        box.addWidget(scroll, 1)
        self.input_checks = {}
        definitions = [
            ("gestures", "Hand gestures", "Move the pointer with your index finger. Hold it still for a left click, or raise index and middle fingers for a right click."),
            ("voice", "Voice commands and dictation", "Dictate text by saying ‘type’ followed by the words. Say ‘space’ or ‘new line’ to format text. Voice commands also support clicking, scrolling, and pausing AccessFlow. Speech is processed locally."),
            ("head", "Head tracking", "Move the pointer with your head. Hold it steady for the selected time to left-click; enable Voice for spoken commands.")
        ]
        for key, title, detail in definitions:
            card = self._card(title, detail)
            check = QCheckBox("Enable")
            card.layout().addWidget(check)
            if key == "head":
                recenter = QPushButton("Recenter head tracking")
                recenter.clicked.connect(self._recenter_head)
                card.layout().addWidget(recenter)
            self.input_checks[key] = check
            content_layout.addWidget(card)

        voice_card = self._card("Offline voice model", "Choose the folder containing an extracted Vosk model, such as vosk-model-small-en-us-0.15.")
        row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Select Vosk model folder")
        browse = QPushButton("Browse")
        browse.setObjectName("secondaryAction")
        browse.setMinimumHeight(42)
        browse.setMinimumWidth(110)
        browse.clicked.connect(self._browse_model)
        self.voice_browse_button = browse
        row.addWidget(self.model_path, 1)
        row.addWidget(browse)
        voice_card.layout().addLayout(row)
        download = QPushButton("Download small English model")
        download.setObjectName("secondaryAction")
        download.setMinimumHeight(42)
        download.setMinimumWidth(220)
        download.clicked.connect(self._download_voice_model)
        self.download_button = download
        voice_card.layout().addWidget(download, 0, Qt.AlignmentFlag.AlignLeft)
        content_layout.addWidget(voice_card)
        content_layout.addStretch()
        return page

    def _recenter_head(self):
        if self.running and self.vision_worker and self.input_checks["head"].isChecked():
            self.vision_worker.recenter_head()
            self.status_label.setText("Head tracking recentered")
        else:
            self.status_label.setText("Start Head tracking first")

    def _calibration(self):
        page, box = self._page("Gaze calibration", "Calibrate nine screen points before eye gaze can control the pointer. Keep your head still and repeat if pointer placement is inaccurate.")
        self.calibration_status = QLabel(self.calibration_notice or "Not calibrated")
        self.calibration_status.setObjectName("notice")
        self.calibration_status.setWordWrap(True)
        box.addWidget(self.calibration_status)
        start = QPushButton("Start calibration")
        start.setObjectName("primary")
        start.clicked.connect(self._begin_calibration)
        box.addWidget(start, 0, Qt.AlignmentFlag.AlignLeft)
        preview = QPushButton("Preview eye mapping")
        preview.clicked.connect(self._preview_gaze)
        box.addWidget(preview, 0, Qt.AlignmentFlag.AlignLeft)
        box.addWidget(self._card(
            "How it works",
            "AccessFlow samples your eye position while you look at each target "
            "and compares affine and quadratic mappings using held-out target "
            "error. It chooses the simpler mapping unless the quadratic fit "
            "meaningfully improves your results. The mapping is saved locally. "
            "Use Tracking preview to check it before desktop use."))
        box.addStretch()
        return page

    def _preview_gaze(self):
        if self.calibration_overlay is not None:
            self._stop_calibration()
        self._navigate(PAGE_PREVIEW)
        self._start_preview("gaze")

    def _begin_calibration(self):
        if self.running:
            self._pause()
        if self.preview_worker is not None:
            self._stop_preview()
        if self._retiring_workers:
            self._pending_calibration = True
            self.calibration_status.setText("Waiting for previous inputs to stop…")
            return
        if not self.input_checks["gaze"].isChecked():
            QMessageBox.information(self, "Select eye gaze", "Enable Eye gaze in Input methods, then start calibration.")
            return
        self.calibration_buffer.clear()
        self.calibration_points = {}
        self.calibration_target_index = 0
        self.calibration_target_started = time.monotonic()
        screen = QApplication.primaryScreen().geometry()
        pointer_width, pointer_height = self.actions.screen_size()
        self.calibration_overlay = CalibrationOverlay(0)
        self.calibration_overlay.setGeometry(screen)
        self.calibration_overlay.showFullScreen()
        self.calibration_overlay.cancelled.connect(self._stop_calibration)
        self.vision_worker = VisionWorker(pointer_width, pointer_height, gaze_enabled=True, gestures_enabled=False)
        self.vision_worker.gaze_sample.connect(self._collect_gaze_sample)
        self.vision_worker.status.connect(self._calibration_status)
        self.vision_worker.start()
        self.calibration_status.setText("Preparing camera and gaze model…")

    def _collect_gaze_sample(self, sample):
        """Buffer eye samples for the current target.

        Only frames from a settled target are kept: the first moments after a
        target appears still carry eye movement from the previous one.
        """
        if not self.calibration_timer.isActive() or self.calibration_overlay is None:
            return
        if time.monotonic() - self.calibration_target_started < CALIBRATION_SAMPLE_GRACE:
            return
        self.calibration_buffer.append((float(sample[0]), float(sample[1])))

    def _poll_calibration(self):
        """Accept the current point once it has settled, then advance or finish."""
        if self.calibration_overlay is None:
            self.calibration_timer.stop()
            return
        elapsed = time.monotonic() - self.calibration_target_started
        samples = np.asarray(self.calibration_buffer, dtype=float)
        if elapsed >= CALIBRATION_MIN_DWELL and len(samples) >= CALIBRATION_MIN_SAMPLES:
            recent = samples[-min(len(samples), 30):]
            spread = float(np.max(np.std(recent, axis=0)))
            if spread <= CALIBRATION_STABLE_STD:
                self.calibration_points[self.calibration_target_index] = (
                    float(np.mean(recent[:, 0])), float(np.mean(recent[:, 1])))
                self.calibration_buffer.clear()
                if self.calibration_target_index < len(self.calibration_overlay.targets) - 1:
                    self.calibration_target_index += 1
                    self.calibration_target_started = time.monotonic()
                    self.calibration_overlay.target_index = self.calibration_target_index
                    self.calibration_overlay.update()
                    self.calibration_status.setText(
                        f"Calibrating point {self.calibration_target_index + 1} of 9…")
                else:
                    self._finish_calibration()
                return
        if elapsed > CALIBRATION_POINT_TIMEOUT:
            if len(samples) < CALIBRATION_MIN_SAMPLES:
                reason = "not enough valid eye readings; check camera framing and eye visibility"
            else:
                recent = samples[-CALIBRATION_MIN_SAMPLES:]
                spread = float(np.max(np.std(recent, axis=0)))
                reason = f"eye reading varied too much ({spread:.3f}); hold your gaze on the dot"
            self.calibration_status.setText(
                f"Calibration stopped at point {self.calibration_target_index + 1}: "
                f"{reason}. Try again with your head still.")
            self._stop_calibration(keep_message=True)

    def _finish_calibration(self):
        self.calibration_timer.stop()
        overlay = self.calibration_overlay
        try:
            targets = overlay.targets if overlay else []
            features, screen_points = [], []
            for index, (target_x, target_y) in enumerate(targets):
                point = self.calibration_points.get(index)
                if point is None:
                    raise ValueError(f"Point {index + 1} was never measured.")
                features.append([point[0], point[1]])
                screen_points.append([target_x, target_y])
            # Prefer the simpler affine map unless leave-one-out validation
            # shows the quadratic meaningfully improves this user's data.
            matrix, error, model, model_errors = fit_best_gaze_mapping(
                features, screen_points)
            if error > CALIBRATION_MAX_ERROR:
                detail = ", ".join(
                    f"{name.removesuffix('_v1')} {value * 100:.0f}%"
                    for name, value in model_errors.items())
                predictions = leave_one_out_gaze_predictions(
                    features, screen_points, model)
                point_details = []
                for index, (feature, target, prediction) in enumerate(
                        zip(features, screen_points, predictions)):
                    predicted = ("unavailable" if prediction is None else
                                 f"{prediction[0]:.2f},{prediction[1]:.2f}")
                    point_details.append(
                        f"P{index + 1} gaze={feature[0]:.3f},{feature[1]:.3f} "
                        f"target={target[0]:.2f},{target[1]:.2f} "
                        f"predicted={predicted}")
                raise ValueError(
                    f"Calibration readings did not form a reliable mapping "
                    f"(held-out errors: {detail}). Keep looking at the center "
                    "of each dot until it advances. Per-point details from "
                    "the chosen fit: " + "; ".join(point_details))
            # Build the mapping completely before adopting it, so a failed
            # attempt never destroys a calibration that already worked.
            matrix_list = matrix.tolist()
            self.settings.setValue(CALIBRATION_KEY, json.dumps({
                "feature": GAZE_FEATURE,
                "model": model,
                "screen": list(self.actions.screen_size()),
                "matrix": matrix_list,
            }))
            self.gaze_transform = matrix_list
            self.calibration_notice = (
                f"Calibrated with {model.removesuffix('_v1')} mapping "
                f"(held-out error {error * 100:.1f}%).")
            self.calibration_status.setText("Calibration saved! You can now start Eye gaze.")
        except Exception as exc:
            kept = " Your previous calibration was kept." if self.gaze_transform else ""
            self.calibration_status.setText(f"Calibration did not finish: {exc}{kept}")
        finally:
            self._stop_calibration(keep_message=True)

    def _calibration_status(self, message):
        # VisionWorker includes the selected camera/backend in its ready
        # message (for example, "Camera inputs active - Camera 0 (DirectShow)").
        # Match the stable prefix so calibration starts sampling after the
        # camera is ready instead of waiting forever for an obsolete exact
        # string. CameraUnavailableError also includes troubleshooting detail,
        # so its status must be matched by prefix as well.
        if message.startswith("Camera inputs active"):
            self.calibration_target_started = time.monotonic()
            self.calibration_status.setText("Calibrating point 1 of 9… look at the red dot")
            self.calibration_timer.start(100)
        elif message.startswith(("Camera stopped", "Camera unavailable")):
            self.calibration_status.setText(message)
            self._stop_calibration(keep_message=True)

    def _stop_calibration(self, keep_message=False):
        self.calibration_timer.stop()
        self.calibration_buffer.clear()
        self._retire_worker(self.vision_worker)
        self.vision_worker = None
        if self.calibration_overlay:
            self.calibration_overlay.close()
            self.calibration_overlay = None
        if not keep_message:
            self.calibration_status.setText("Calibration cancelled")

    def _load_calibration(self):
        """Restore a saved gaze mapping, rejecting one made for another screen."""
        saved = self.settings.value(CALIBRATION_KEY, "")
        if not saved:
            # Mapping saved by an earlier version, before screen size was stored.
            saved = self.settings.value("gaze/transform_v3", "")
        if not saved:
            return None
        try:
            data = json.loads(saved)
        except (TypeError, ValueError):
            self.calibration_notice = "Saved calibration is unreadable; recalibrate."
            return None
        if isinstance(data, list):
            # Mapping saved before the screen size and feature were recorded.
            matrix = _valid_gaze_matrix(data)
            if matrix is None:
                self.calibration_notice = "Saved calibration is unreadable; recalibrate."
                return None
            self.calibration_notice = "Recalibrate to record your screen size with the mapping."
            return matrix
        if not isinstance(data, dict) or data.get("feature") != GAZE_FEATURE:
            self.calibration_notice = "Saved calibration used an older gaze feature; recalibrate."
            return None
        recorded = data.get("screen")
        if recorded:
            try:
                recorded_size = list(recorded)
            except TypeError:
                self.calibration_notice = "Saved calibration screen size is unreadable; recalibrate."
                return None
            if recorded_size != list(self.actions.screen_size()):
                self.calibration_notice = (
                    "Saved calibration was made for a different screen size; recalibrate.")
                return None
        if not recorded:
            self.calibration_notice = (
                "Saved calibration has no screen-size record; recalibrate to verify it.")
        matrix = _valid_gaze_matrix(data.get("matrix"))
        if matrix is None:
            self.calibration_notice = "Saved calibration is unreadable; recalibrate."
        else:
            matrix_model = model_for_matrix(matrix)
            stored_model = data.get("model")
            if stored_model is None:
                self.calibration_notice = (
                    f"Saved eye calibration loaded as {matrix_model.removesuffix('_v1')}; "
                    "model metadata was not recorded.")
            elif stored_model not in (GAZE_MODEL_AFFINE, GAZE_MODEL_QUADRATIC):
                self.calibration_notice = "Saved calibration model is unsupported; recalibrate."
                return None
            elif stored_model != matrix_model:
                self.calibration_notice = "Saved calibration model is inconsistent; recalibrate."
                return None
            elif not self.calibration_notice:
                self.calibration_notice = (
                    f"Saved {stored_model.removesuffix('_v1')} eye calibration loaded.")
        return matrix

    def _preview(self):
        page, box = self._page(
            "Tracking preview",
            "See what the camera sees and exactly where each input method sends "
            "the pointer. Preview never moves the real pointer, so it is safe to "
            "experiment before starting a session.")
        row = QHBoxLayout()
        for mode, label in (("hand", "Hand preview"), ("head", "Head preview")):
            button = QPushButton(label)
            button.clicked.connect(
                lambda checked=False, m=mode: self._start_preview(m))
            row.addWidget(button)
        stop = QPushButton("Stop preview")
        stop.clicked.connect(self._stop_preview)
        row.addWidget(stop)
        row.addStretch()
        box.addLayout(row)

        self.preview_widget = TrackingPreviewWidget()
        box.addWidget(self.preview_widget, 1)

        self.preview_status = QLabel("Idle")
        self.preview_status.setObjectName("notice")
        self.preview_status.setWordWrap(True)
        box.addWidget(self.preview_status)

        self.preview_hint = QLabel("")
        self.preview_hint.setObjectName("subtitle")
        self.preview_hint.setWordWrap(True)
        box.addWidget(self.preview_hint)
        return page

    def _start_preview(self, mode):
        if mode not in ("hand", "head"):
            return
        if self.running:
            self._pause()
        if self.preview_worker is not None:
            self._stop_preview()
        if self._retiring_workers:
            self._pending_preview = mode
            self.preview_status.setText("Waiting for the previous camera session to stop…")
            return
        pointer_width, pointer_height = self.actions.screen_size()
        self.preview_worker = VisionWorker(
            pointer_width, pointer_height,
            gaze_enabled=False,
            gestures_enabled=(mode == "hand"),
            head_enabled=(mode == "head"),
            preview=mode,
            dwell_ms=self.dwell_slider.value())
        self.preview_worker.preview_ready.connect(self.preview_widget.set_payload)
        self.preview_worker.status.connect(self._preview_status)
        self.preview_worker.finished.connect(self._on_worker_finished)
        self.preview_mode = mode
        self.preview_widget.clear()
        self.preview_worker.start()
        hints = {
            "hand": ("Move your index finger like a trackpad. The mapping is "
                     "relative: the cursor moves with the finger's movement, not "
                     "to where the finger points. This preview checks pointer "
                     "movement only and never arms or performs clicks. In a live "
                     "session, keep only your index fingertip still for left "
                     "click, or hold only index and middle fingers up for right "
                     "click. Pinching does not click."),
            "head": ("Move your head left/right/up/down. The first frames learn "
                     "your neutral position; stay still while the preview "
                     "settles. Hold the pointer within a small area for the "
                     "selected dwell time to left-click. The preview timer never "
                     "clicks."),
        }
        self.preview_hint.setText(hints[mode])
        self.preview_status.setText("Starting camera…")

    def _preview_status(self, message):
        self.preview_status.setText(message)

    def _stop_preview(self):
        worker = self.preview_worker
        self.preview_worker = None
        self.preview_mode = None
        if hasattr(self, "preview_widget"):
            self.preview_widget.clear()
        if worker is not None:
            self._retire_worker(worker)
        if hasattr(self, "preview_status"):
            self.preview_status.setText("Preview stopped")

    def _download_voice_model(self):
        if self.download_worker and self.download_worker.isRunning():
            return
        self.download_worker = VoiceModelDownloadWorker()
        self.download_worker.status.connect(self.status_label.setText)
        self.download_worker.completed.connect(self._voice_model_downloaded)
        self.download_worker.failed.connect(self._voice_model_download_failed)
        self.download_button.setEnabled(False)
        self.download_worker.start()

    def _voice_model_downloaded(self, path):
        self.download_button.setEnabled(not self.running)
        self.model_path.setText(path)
        self.status_label.setText("Offline voice model ready")

    def _voice_model_download_failed(self, message):
        self.download_button.setEnabled(not self.running)
        QMessageBox.warning(self, "Model download failed", message)

    def _accessibility(self):
        page, box = self._page("Accessibility", "Settings are stored locally on this computer.")
        keyboard = self._card(
            "On-screen keyboard",
            "Select a text field in the app you want to type into, then dwell on keys in AccessFlow’s keyboard.")
        open_keyboard = QPushButton("Open AccessFlow Keyboard")
        open_keyboard.setObjectName("secondaryAction")
        open_keyboard.setMinimumHeight(42)
        open_keyboard.clicked.connect(self._show_accessflow_keyboard)
        keyboard.layout().addWidget(open_keyboard)
        box.addWidget(keyboard)
        dwell = self._card("Dwell selection", "Hand: hold your index finger still for a left click, or raise index and middle fingers for a right click. Head: hold the pointer steady for a left click. Choose how long to dwell below. Voice can also perform left and right clicks.")
        value = QLabel("1200 ms")
        self.dwell_slider = QSlider(Qt.Orientation.Horizontal)
        self.dwell_slider.setRange(500, 3000)
        self.dwell_slider.setSingleStep(100)
        self.dwell_slider.setValue(1200)
        self.dwell_slider.valueChanged.connect(lambda ms: value.setText(f"{ms} ms"))
        dwell.layout().addWidget(value)
        dwell.layout().addWidget(self.dwell_slider)
        box.addWidget(dwell)
        box.addWidget(self._card("Emergency pause", "Use the dashboard button, the system tray, or Escape while AccessFlow is open."))
        box.addWidget(self._card("Privacy", "Webcam frames and microphone audio are processed locally and are not recorded by AccessFlow."))
        box.addStretch()
        return page

    def _show_accessflow_keyboard(self):
        keyboard = self.accessflow_keyboard
        screen = QApplication.primaryScreen().availableGeometry()
        keyboard.adjustSize()
        keyboard.move(
            screen.left() + max(0, (screen.width() - keyboard.width()) // 2),
            screen.bottom() - keyboard.height() - 24)
        keyboard.show()
        keyboard.raise_()
        self.status_label.setText(
            "AccessFlow keyboard ready · select a text field, then choose keys")

    def _keyboard_key_requested(self, value, is_key):
        # The on-screen keyboard is also a direct mouse/touchpad input method;
        # typing must not depend on starting the camera or voice session.
        if is_key:
            self.action_executor.submit(self.actions.press_key, {"key": value})
        else:
            self.action_executor.submit(self.actions.type_text, {"text": value})

    def _show_dwell_feedback(self, payload):
        screen = QApplication.primaryScreen()
        scale = screen.devicePixelRatio() if screen is not None else 1.0
        self.dwell_feedback_overlay.show_progress(
            payload.get("point", (0, 0)), payload.get("progress", 0.0), scale)

    def _about(self):
        page, box = self._page(
            "About AccessFlow",
            "An accessibility app for pointer control, dwell selection, voice input, and on-screen typing.")
        box.addWidget(self._card(
            "Basic features",
            "Use hand or head movement to control the pointer, dwell to click, "
            "speak commands or dictate text, and type with the built-in keyboard."))
        box.addStretch()
        return page

    def _navigate(self, index):
        if index != PAGE_PREVIEW and getattr(self, "preview_worker", None) is not None:
            self._stop_preview()
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)

    def _build_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))
        self.tray.setToolTip("AccessFlow")
        menu = QMenu()
        show = QAction("Open AccessFlow", self)
        show.triggered.connect(self._show_from_tray)
        toggle = QAction("Start / pause inputs", self)
        toggle.triggered.connect(self._toggle_running)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self._quit)
        menu.addAction(show)
        menu.addAction(toggle)
        menu.addSeparator()
        menu.addAction(exit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _show_from_tray(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self._show_from_tray()

    def _restore_settings(self):
        for key, check in self.input_checks.items():
            check.setChecked(self.settings.value(f"inputs/{key}", False, type=bool))
            check.toggled.connect(lambda value, name=key: self.settings.setValue(f"inputs/{name}", value))
        self.model_path.setText(self.settings.value("voice/model_path", ""))
        self.model_path.textChanged.connect(lambda value: self.settings.setValue("voice/model_path", value))
        self.dwell_slider.setValue(self.settings.value("accessibility/dwell_ms", 1200, type=int))
        self.dwell_slider.valueChanged.connect(lambda value: self.settings.setValue("accessibility/dwell_ms", value))

    def _browse_model(self):
        start_dir = self.model_path.text().strip()
        if not Path(start_dir).is_dir():
            start_dir = str(Path.home())
        dialog = QFileDialog(self, "Select extracted Vosk model folder", start_dir)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        # Avoid native shell dialogs, which can block the Qt event loop on
        # systems where the shell's folder picker hangs or fails to initialize.
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                self.model_path.setText(selected[0])

    def _toggle_running(self):
        if self.running:
            self._pause()
        else:
            self._start()

    def _start(self):
        # A preview owns the camera until it is stopped.
        if self.preview_worker is not None:
            self._stop_preview()
        if self._retiring_workers:
            self._pending_start = True
            self.status_label.setText("Waiting for previous inputs to stop…")
            return
        gestures = self.input_checks["gestures"].isChecked()
        voice = self.input_checks["voice"].isChecked()
        head = self.input_checks["head"].isChecked()
        if not any((gestures, voice, head)):
            QMessageBox.information(
                self, "Select an input method",
                "Enable at least one input method before starting a session.")
            return
        model_path = Path(self.model_path.text().strip())
        voice_skipped = voice and not model_path.is_dir()
        if voice_skipped:
            voice = False
        sources = []
        if gestures:
            sources.append("gestures")
        if voice:
            sources.append("voice")
        if head:
            sources.append("head")
        notices = []
        if voice_skipped:
            notices.append("Voice needs an extracted Vosk model folder")
        if not sources:
            if notices:
                QMessageBox.information(
                    self, "Input setup needed", "; ".join(notices) + ".")
                return
            QMessageBox.information(
                self, "Select an input method",
                "No usable input method is selected. Enable hand, head, or voice input.")
            return
        if notices:
            QMessageBox.information(
                self, "Some inputs will be skipped",
                "; ".join(notices) + ". Other selected methods will start.")
        # Each selected provider starts independently; camera tracking serves
        # the selected head and/or hand inputs through one worker.
        self._active = True
        self.engine.start(sources)
        self.running = True
        self._set_input_controls_enabled(False)
        self.start_button.setText("Pause inputs")
        self.status_label.setText("Active")
        self.session_card.layout().itemAt(0).widget().setText("Session active")
        source_summary = ", ".join(sources)
        if voice_skipped:
            source_summary += " · Voice skipped (model required)"
        self.session_card.layout().itemAt(1).widget().setText(source_summary)

        if gestures or head:
            pointer_width, pointer_height = self.actions.screen_size()
            self.vision_worker = VisionWorker(
                pointer_width, pointer_height, gaze_enabled=False,
                gestures_enabled=gestures, head_enabled=head,
                dwell_ms=self.dwell_slider.value())
            self.vision_worker.event_ready.connect(self._on_event)
            self.vision_worker.status.connect(self.status_label.setText)
            self.vision_worker.dwell_feedback.connect(self._show_dwell_feedback)
            self.vision_worker.start()
        if voice:
            self.voice_worker = VoiceWorker(str(model_path))
            self.voice_worker.event_ready.connect(self._on_event)
            self.voice_worker.status.connect(self.status_label.setText)
            self.voice_worker.start()

    def _set_input_controls_enabled(self, enabled):
        """Keep displayed settings aligned with the running worker settings."""
        for check in self.input_checks.values():
            check.setEnabled(enabled)
        self.model_path.setEnabled(enabled)
        self.voice_browse_button.setEnabled(enabled)
        self.download_button.setEnabled(
            enabled and not (self.download_worker and self.download_worker.isRunning()))
        self.dwell_slider.setEnabled(enabled)

    def _on_event(self, event):
        if event.action == "pause":
            self._pause()
            return
        if event.action == "move_pointer":
            if not self.running:
                return
            # Camera workers can emit a move every frame. Keep only the newest
            # point and apply it on a short UI timer instead of replaying an
            # increasingly stale trail if the GUI thread falls behind.
            self._pending_pointer_event = event
            if not self.pointer_move_timer.isActive():
                self.pointer_move_timer.start()
            return
        if self._pending_pointer_event is not None:
            # Apply the latest position before clicks or other actions that
            # depend on where the cursor currently is.
            self.pointer_move_timer.stop()
            self._flush_pointer_move()
        self.engine.dispatch(event)

    def _flush_pointer_move(self):
        event = self._pending_pointer_event
        self._pending_pointer_event = None
        if event is not None and self.running:
            self.engine.dispatch(event)

    def _report_error(self, message):
        """Show a failed action instead of discarding it silently.

        Runs on whatever thread performed the action, so the message is handed
        to the UI thread through the bus signal rather than touching a widget.
        """
        self.bus.status.emit(message)

    def _retire_worker(self, worker):
        """Stop a worker without blocking the GUI thread."""
        if worker is None:
            return
        # Drop UI callbacks first so late status/event signals cannot overwrite
        # the Paused state while the worker tears down.
        with warnings.catch_warnings():
            # PySide warns when disconnecting a signal that has no listeners.
            warnings.simplefilter("ignore", RuntimeWarning)
            for name in ("status", "event_ready", "gaze_sample", "preview_ready",
                         "dwell_feedback", "completed", "failed"):
                signal = getattr(worker, name, None)
                if signal is not None:
                    try:
                        signal.disconnect()
                    except (RuntimeError, TypeError):
                        pass
        worker.stop()
        worker.finished.connect(self._on_worker_finished)
        if worker.isRunning():
            self._retiring_workers.append(worker)
        else:
            worker.deleteLater()

    def _on_worker_finished(self):
        worker = self.sender()
        if worker in self._retiring_workers:
            self._retiring_workers.remove(worker)
        if worker is not None:
            if worker is self.preview_worker:
                self.preview_worker = None
                self.preview_mode = None
            worker.deleteLater()
        if self._retiring_workers:
            return
        if self._quitting:
            self._quitting = False
            QApplication.quit()
        elif self._pending_start:
            self._pending_start = False
            QTimer.singleShot(0, self._start)
        elif self._pending_calibration:
            self._pending_calibration = False
            QTimer.singleShot(0, self._begin_calibration)
        elif self._pending_preview:
            mode = self._pending_preview
            self._pending_preview = None
            QTimer.singleShot(0, lambda: self._start_preview(mode))

    def _pause_workers(self):
        if self.calibration_timer.isActive():
            self._stop_calibration()
        self._stop_preview()
        self._retire_worker(self.vision_worker)
        self._retire_worker(self.voice_worker)
        self.vision_worker = None
        self.voice_worker = None

    def _pause(self):
        # Update visible state first: stopping workers must never delay feedback.
        self._pending_start = False
        self._pending_calibration = False
        self._pending_preview = None
        self._pending_pointer_event = None
        self.pointer_move_timer.stop()
        self.dwell_feedback_overlay.hide_timer.stop()
        self.dwell_feedback_overlay.hide()
        self.running = False
        self._set_input_controls_enabled(True)
        self.start_button.setText("Start selected inputs")
        self.status_label.setText("Paused")
        self.session_card.layout().itemAt(0).widget().setText("Session paused")
        self.session_card.layout().itemAt(1).widget().setText("Camera and microphone are off.")
        self.engine.pause()
        self._active = False

    def _active_input(self, key):
        """Return True when an input method is enabled for the session."""
        if key == "head":
            return self.input_checks["head"].isChecked()
        if key == "gestures":
            return self.input_checks["gestures"].isChecked()
        if key == "voice":
            return self.input_checks["voice"].isChecked()
        return False

    def _hook_failed(self, message):
        """Warn when a global hook could not be installed.

        Installing a low-level hook can fail (policy, exhausted hook quotas, a
        different desktop session). Running silently would leave the user
        believing Escape is always available, so say what still works.
        """
        self.safety_warning = message
        self.status_label.setText(message)
        note = getattr(self, "dashboard_note", None)
        if note is not None:
            note.setText(f"Safety warning: {message}")
        if self.tray.isVisible():
            self.tray.showMessage("AccessFlow safety", message,
                                  QSystemTrayIcon.MessageIcon.Warning, 10000)

    def _global_escape(self):
        # The hook does not consume Escape, so the focused app still gets it.
        if (self.running or self._pending_start or self._pending_calibration
                or self._pending_preview or self.calibration_overlay is not None
                or self.preview_worker is not None):
            self._pause()

    def _touchpad_takeover(self):
        # Real mouse/touchpad movement: let the camera pointer step aside briefly.
        if self.running and self.vision_worker is not None:
            self.vision_worker.suspend_pointer()
            self.dwell_feedback_overlay.hide_timer.stop()
            self.dwell_feedback_overlay.hide()

    def _stop_global_listeners(self):
        for listener in (self.esc_listener, self.pointer_listener):
            if listener.isRunning():
                listener.stop()

    def _quit(self):
        # Stop observing input before any deferred shutdown begins.
        self._stop_global_listeners()
        self._pause()
        if self.download_worker is not None:
            worker = self.download_worker
            self.download_worker = None
            self._retire_worker(worker)
        # _pause() cancels queued work; joining here keeps the process from
        # exiting halfway through a keystroke (which can leave a modifier held).
        self.action_executor.stop()
        if self._retiring_workers:
            self._quitting = True
            self.hide()
            QTimer.singleShot(8000, self._force_quit)
        else:
            QApplication.quit()

    def _force_quit(self):
        if not self._quitting:
            return
        self._quitting = False
        QApplication.quit()

    def closeEvent(self, event):
        if self.tray.isVisible():
            self._pause()
            self.accessflow_keyboard.hide()
            self.dwell_feedback_overlay.hide()
            self.hide()
            event.ignore()
        else:
            self._quit()
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet("""
        QWidget { font-family: 'Segoe UI'; font-size: 14px; color: #172536; background: #f5f7fa; }
        #sidebar { background: #102238; }
        #sidebar QLabel { color: #b7c5d2; font-size: 10px; letter-spacing: 1px; }
        #brand { color: white; font-size: 25px; font-weight: 700; letter-spacing: 0; }
        #nav { color: #dbe5ee; text-align: left; padding: 12px 14px; border: 0; border-radius: 8px; background: transparent; }
        #nav:hover, #nav:checked { background: #23415f; color: white; }
        #sideStatus { color: #a7f3d0; font-size: 13px; letter-spacing: 0; }
        #heading { font-size: 30px; font-weight: 700; color: #14283d; }
        #subtitle { font-size: 15px; color: #5e7185; }
        #card { background: white; border: 1px solid #e1e8ef; border-radius: 12px; }
        #card QLabel { background: transparent; }
        #cardTitle { font-size: 16px; font-weight: 650; color: #203950; }
        #cardText { color: #617487; }
        QPushButton { padding: 10px 16px; border: 1px solid #ccd7e2; border-radius: 8px; background: white; }
        QPushButton:hover { background: #edf4fa; }
        QPushButton:checked { background: #d8e8f4; border-color: #8faec7; }
        QPushButton#secondaryAction { background: #e8f1f8; color: #163c5b; border: 1px solid #b8cad9; font-weight: 600; }
        QPushButton#secondaryAction:hover { background: #d8e8f4; }
        #primary { background: #1769aa; color: white; border: 0; font-weight: 650; }
        #primary:hover { background: #0f568f; }
        #notice { background: #eaf3fb; color: #315978; padding: 14px; border-radius: 8px; }
    """)
    window = AccessFlowWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

