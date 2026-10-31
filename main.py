from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QSettings, Signal, QObject, QTimer, QRect
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QColor, QPainter
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton, QSlider,
    QStackedWidget, QSystemTrayIcon, QVBoxLayout, QWidget
)

from accessflow.core.engine import InteractionEngine
from accessflow.system.actions import WindowsActions
from accessflow.inputs.vision import VisionWorker
from accessflow.inputs.voice import VoiceModelDownloadWorker, VoiceWorker

APP_NAME = "AccessFlow"


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

class Bus(QObject):
    status = Signal(str)


class AccessFlowWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("Vidushi", APP_NAME)
        self.bus = Bus()
        self.actions = WindowsActions()
        self.engine = InteractionEngine(self.actions, on_pause=self._pause_workers)
        self.vision_worker = None
        self.voice_worker = None
        self.running = False
        self.gaze_transform = None
        saved_transform = self.settings.value("gaze/transform_v3", "")
        if saved_transform:
            try:
                self.gaze_transform = json.loads(saved_transform)
            except (TypeError, ValueError):
                self.gaze_transform = None
        self.calibration_samples = []
        self.calibration_target_index = 0
        self.calibration_target_started = 0.0
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self._next_calibration_target)
        self.calibration_overlay = None
        self.download_worker = None
        self.setWindowTitle("AccessFlow")
        self.resize(1120, 740)
        self.setMinimumSize(920, 620)
        self._build_ui()
        self._build_tray()
        self._restore_settings()
        QShortcut(QKeySequence("Esc"), self, activated=self._pause)

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
        for index, name in enumerate(["Dashboard", "Input methods", "Calibration", "Accessibility", "About"]):
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

        for page in (self._dashboard(), self._inputs(), self._calibration(), self._accessibility(), self._about()):
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

        note = QLabel("Camera tracking combines gaze and hand input through one webcam owner. Voice recognition is local and needs an extracted Vosk model folder. Inputs start only when selected here.")
        note.setObjectName("notice")
        note.setWordWrap(True)
        box.addWidget(note)
        box.addStretch()
        return page

    def _inputs(self):
        page, box = self._page("Input methods", "Select what AccessFlow should listen to when you start a session.")
        self.input_checks = {}
        definitions = [
            ("gaze", "Eye gaze", "Moves the pointer from eye direction. First-run face model downloads from MediaPipe."),
            ("gestures", "Hand gestures", "Point to move and pinch to click. Shares the camera worker with gaze."),
            ("voice", "Voice commands and dictation", "Offline Vosk recognition; audio is processed locally."),
        ]
        for key, title, detail in definitions:
            card = self._card(title, detail)
            check = QCheckBox("Enable")
            card.layout().addWidget(check)
            self.input_checks[key] = check
            box.addWidget(card)

        voice_card = self._card("Offline voice model", "Choose the folder containing an extracted Vosk model, such as vosk-model-small-en-us-0.15.")
        row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Select Vosk model folder")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_model)
        row.addWidget(self.model_path, 1)
        row.addWidget(browse)
        voice_card.layout().addLayout(row)
        download = QPushButton("Download small English model")
        download.clicked.connect(self._download_voice_model)
        voice_card.layout().addWidget(download)
        box.addWidget(voice_card)
        box.addStretch()
        return page

    def _calibration(self):
        page, box = self._page("Gaze calibration", "Calibrate nine screen points before eye gaze can control the pointer.")
        self.calibration_status = QLabel("Not calibrated")
        self.calibration_status.setObjectName("notice")
        self.calibration_status.setWordWrap(True)
        box.addWidget(self.calibration_status)
        start = QPushButton("Start nine-point calibration")
        start.setObjectName("primary")
        start.clicked.connect(self._begin_calibration)
        box.addWidget(start, 0, Qt.AlignmentFlag.AlignLeft)
        box.addWidget(self._card("How it works", "AccessFlow samples your eye position while you look at each target. The mapping is saved locally. Keep your head still and repeat if pointer placement is inaccurate."))
        box.addStretch()
        return page

    def _begin_calibration(self):
        if self.running:
            self._pause()
        if not self.input_checks["gaze"].isChecked():
            QMessageBox.information(self, "Select eye gaze", "Enable Eye gaze in Input methods, then start calibration.")
            return
        self.calibration_samples = []
        self.calibration_target_index = 0
        self.calibration_target_started = time.monotonic()
        screen = QApplication.primaryScreen().geometry()
        pointer_width, pointer_height = self.actions.screen_size()
        self.calibration_overlay = CalibrationOverlay(0)
        self.calibration_overlay.setGeometry(screen)
        self.calibration_overlay.showFullScreen()
        self.calibration_overlay.cancelled.connect(self._stop_calibration)
        self.calibration_overlay.installEventFilter(self)
        self.vision_worker = VisionWorker(pointer_width, pointer_height, True, False)
        self.vision_worker.gaze_sample.connect(self._collect_gaze_sample)
        self.vision_worker.status.connect(self._calibration_status)
        self.vision_worker.start()
        self.calibration_status.setText("Preparing camera and gaze model…")

    def _collect_gaze_sample(self, sample):
        if not self.calibration_timer.isActive() or time.monotonic() - self.calibration_target_started < 0.25:
            return
        self.calibration_samples.append((self.calibration_target_index, float(sample[0]), float(sample[1])))

    def _next_calibration_target(self):
        if self.calibration_target_index < 8:
            self.calibration_target_index += 1
            self.calibration_target_started = time.monotonic()
            self.calibration_overlay.target_index = self.calibration_target_index
            self.calibration_overlay.update()
            self.calibration_status.setText(f"Calibrating point {self.calibration_target_index + 1} of 9…")
            self.calibration_timer.start(1800)
            return
        self.calibration_timer.stop()
        try:
            import numpy as np
            features, targets = [], []
            for target_index, (target_x, target_y) in enumerate(self.calibration_overlay.targets):
                samples = [(rx, ry) for index, rx, ry in self.calibration_samples if index == target_index]
                if len(samples) < 10:
                    raise ValueError(f"Too few eye samples at point {target_index + 1}. Check camera framing and try again.")
                if float(np.max(np.std(np.asarray(samples), axis=0))) > 0.07:
                    raise ValueError(f"Eye tracking was unsteady at point {target_index + 1}. Hold still and try that calibration again.")
                center = np.median(np.asarray(samples), axis=0)
                features.append([float(center[0]), float(center[1]), 1.0])
                targets.append([target_x, target_y])
            x = np.asarray(features)
            y = np.asarray(targets)
            matrix, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
            if rank < 3 or float(np.max(np.abs(matrix[:2, :]))) > 14.0:
                raise ValueError("Eye movement was too small or unstable to calibrate. Keep the camera at eye level, look at each dot, and try again.")
            error = float(np.sqrt(np.mean((x @ matrix - y) ** 2)))
            if error > 0.16:
                raise ValueError("Calibration readings varied too much. Improve lighting, keep your head still, and try again.")
            self.gaze_transform = matrix.tolist()
            self.settings.setValue("gaze/transform_v3", json.dumps(self.gaze_transform))
            self.calibration_status.setText("Calibration saved. Start with a small gaze movement; recalibrate if pointer placement is off.")
        except Exception as exc:
            self.gaze_transform = None
            self.settings.remove("gaze/transform_v3")
            self.calibration_status.setText(f"Calibration did not finish: {exc}")
        finally:
            self._stop_calibration()
    def _calibration_status(self, message):
        if message == "Camera inputs active":
            self.calibration_target_started = time.monotonic()
            self.calibration_status.setText("Calibrating point 1 of 9…")
            self.calibration_timer.start(2000)
        elif message.startswith("Camera stopped") or message == "Camera unavailable":
            self.calibration_timer.stop()
            self.calibration_status.setText(message)
            self._stop_calibration()

    def _stop_calibration(self):
        self.calibration_timer.stop()
        if self.vision_worker and self.vision_worker.isRunning():
            self.vision_worker.stop()
        self.vision_worker = None
        if self.calibration_overlay:
            self.calibration_overlay.close()
            self.calibration_overlay = None

    def _calibration_cancelled(self, event):
        self._stop_calibration()
        event.accept()

    def _download_voice_model(self):
        if self.download_worker and self.download_worker.isRunning():
            return
        self.download_worker = VoiceModelDownloadWorker()
        self.download_worker.status.connect(self.status_label.setText)
        self.download_worker.completed.connect(self._voice_model_downloaded)
        self.download_worker.failed.connect(lambda message: QMessageBox.warning(self, "Model download failed", message))
        self.download_worker.start()

    def _voice_model_downloaded(self, path):
        self.model_path.setText(path)
        self.status_label.setText("Offline voice model ready")

    def _accessibility(self):
        page, box = self._page("Accessibility", "Settings are stored locally on this computer.")
        dwell = self._card("Dwell duration", "Target dwell time for future gaze selection.")
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

    def _about(self):
        page, box = self._page("About AccessFlow", "Adaptive multimodal assistive computer interaction.")
        box.addWidget(self._card("Interaction engine", "Every provider emits a typed event. The coordinator checks session state, source, and confidence before routing desktop actions."))
        box.addWidget(self._card("Windows integration", "PyAutoGUI handles pointer and keyboard actions; pywinauto supports named UI controls and window activation. Application launching is restricted to Notepad and Calculator."))
        box.addStretch()
        return page

    def _navigate(self, index):
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)

    def _build_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))
        self.tray.setToolTip("AccessFlow")
        menu = QMenu()
        show = QAction("Open AccessFlow", self)
        show.triggered.connect(self.showNormal)
        toggle = QAction("Start / pause inputs", self)
        toggle.triggered.connect(self._toggle_running)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.quit)
        menu.addAction(show)
        menu.addAction(toggle)
        menu.addSeparator()
        menu.addAction(exit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.showNormal() if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _restore_settings(self):
        for key, check in self.input_checks.items():
            check.setChecked(self.settings.value(f"inputs/{key}", False, type=bool))
            check.toggled.connect(lambda value, name=key: self.settings.setValue(f"inputs/{name}", value))
        self.model_path.setText(self.settings.value("voice/model_path", ""))
        self.model_path.textChanged.connect(lambda value: self.settings.setValue("voice/model_path", value))
        self.dwell_slider.setValue(self.settings.value("accessibility/dwell_ms", 1200, type=int))
        self.dwell_slider.valueChanged.connect(lambda value: self.settings.setValue("accessibility/dwell_ms", value))

    def _browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select extracted Vosk model folder")
        if path:
            self.model_path.setText(path)

    def _toggle_running(self):
        if self.running:
            self._pause()
        else:
            self._start()

    def _start(self):
        gaze = self.input_checks["gaze"].isChecked()
        gestures = self.input_checks["gestures"].isChecked()
        voice = self.input_checks["voice"].isChecked()
        model_path = Path(self.model_path.text().strip())
        if voice and not model_path.is_dir():
            QMessageBox.warning(self, "Vosk model required", "Select an extracted Vosk model folder before enabling voice input.")
            return
        if gaze and not self.gaze_transform:
            QMessageBox.warning(self, "Calibrate gaze first", "Open Calibration and complete the nine-point calibration before starting eye gaze.")
            self._navigate(2)
            return
        sources = []
        if gaze:
            sources.append("gaze")
        if gestures:
            sources.append("gestures")
        if voice:
            sources.append("voice")
        self.engine.start(sources)
        self.running = True
        self.start_button.setText("Pause inputs")
        self.status_label.setText("Active" if sources else "Active · no inputs")
        self.session_card.layout().itemAt(0).widget().setText("Session active")
        self.session_card.layout().itemAt(1).widget().setText(", ".join(sources) if sources else "No input methods selected")

        if gaze or gestures:
            pointer_width, pointer_height = self.actions.screen_size()
            self.vision_worker = VisionWorker(pointer_width, pointer_height, gaze, gestures, self.gaze_transform)
            self.vision_worker.event_ready.connect(self._on_event)
            self.vision_worker.status.connect(self.status_label.setText)
            self.vision_worker.start()
        if voice:
            self.voice_worker = VoiceWorker(str(model_path))
            self.voice_worker.event_ready.connect(self._on_event)
            self.voice_worker.status.connect(self.status_label.setText)
            self.voice_worker.start()

    def _on_event(self, event):
        if event.action == "pause":
            self._pause()
            return
        self.engine.dispatch(event)

    def _pause_workers(self):
        self._stop_calibration() if self.calibration_timer.isActive() else None
        for worker in (self.vision_worker, self.voice_worker):
            if worker and worker.isRunning():
                worker.stop()
        self.vision_worker = None
        self.voice_worker = None

    def _pause(self):
        self.engine.pause()
        self.running = False
        self.start_button.setText("Start selected inputs")
        self.status_label.setText("Paused")
        self.session_card.layout().itemAt(0).widget().setText("Session paused")
        self.session_card.layout().itemAt(1).widget().setText("Camera and microphone are off.")
    def closeEvent(self, event):
        if self.tray.isVisible():
            self._pause()
            self.hide()
            event.ignore()
        else:
            self._pause()
            event.accept()
            QApplication.quit()


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
        #primary { background: #1769aa; color: white; border: 0; font-weight: 650; }
        #primary:hover { background: #0f568f; }
        #notice { background: #eaf3fb; color: #315978; padding: 14px; border-radius: 8px; }
    """)
    window = AccessFlowWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

