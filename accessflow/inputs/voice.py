import json
import math
import os
import queue
import urllib.request
import zipfile
from pathlib import Path

import sounddevice as sd
from PySide6.QtCore import QThread, Signal
from vosk import Model, KaldiRecognizer
from accessflow.core.events import InteractionEvent

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

# Commands act on the desktop, so the weakest word must be heard clearly.
COMMAND_MIN_CONFIDENCE = 0.72
# Dictation is scored on the average word; keeps the two thresholds together.
DICTATION_MIN_CONFIDENCE = 0.60
# The app-launch vocabulary is an exact, allow-listed two-phrase command.
# Average confidence is more reliable than rejecting the phrase for one weak
# word, while the whitelist still prevents arbitrary program execution.
OPEN_APP_MIN_CONFIDENCE = 0.60
KEY_COMMAND_MIN_CONFIDENCE = 0.60
# A correct left-click command can have one weakly scored word (often "left")
# even when the phrase as a whole is clear. Keep an exact-phrase check and use
# average word confidence, while leaving destructive/open/scroll commands at
# the stricter weakest-word threshold.
LEFT_CLICK_MIN_CONFIDENCE = 0.65
# Blocks waiting for the recogniser (8000 frames at 16 kHz = 0.5 s each).
# Recognition slower than realtime must drop audio, not grow without bound.
AUDIO_QUEUE_BLOCKS = 50


class VoiceModelDownloadWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.stop_requested = False

    def stop(self):
        """Stop a model download between bounded network reads."""
        self.stop_requested = True

    def run(self):
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AccessFlow" / "models"
        archive = base / "vosk-model-small-en-us-0.15.zip"
        temporary = archive.with_suffix(".download")
        try:
            base.mkdir(parents=True, exist_ok=True)
            self.status.emit("Downloading offline speech model…")
            request = urllib.request.Request(
                MODEL_URL, headers={"User-Agent": "AccessFlow/1.0"})
            with urllib.request.urlopen(request, timeout=5) as response, \
                    temporary.open("wb") as output:
                total = int(response.headers.get("Content-Length", 0) or 0)
                received = 0
                while not self.stop_requested:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    if total:
                        self.status.emit(
                            f"Downloading offline speech model… {received * 100 // total}%")
            if self.stop_requested:
                return
            temporary.replace(archive)
            with zipfile.ZipFile(archive) as zipped:
                for info in zipped.infolist():
                    target = (base / info.filename).resolve()
                    if os.path.commonpath((str(base.resolve()), str(target))) != str(base.resolve()):
                        raise ValueError("Model archive contains an unsafe path")
                zipped.extractall(base)
            if self.stop_requested:
                return
            model = base / "vosk-model-small-en-us-0.15"
            self.completed.emit(str(model))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)


class VoiceWorker(QThread):
    event_ready = Signal(object)
    status = Signal(str)

    def __init__(self, model_directory):
        super().__init__()
        self.model_directory = model_directory
        self.stop_requested = False
        self.audio = queue.Queue(maxsize=AUDIO_QUEUE_BLOCKS)
        self.dropped_blocks = 0
        self.reported_backlog = False

    def stop(self):
        """Ask the worker to stop without blocking the caller.

        The stream is closed on the worker thread; callers keep the worker alive
        until its ``finished`` signal arrives.
        """
        self.stop_requested = True

    def _phrase(self, value, confidence, average=None):
        recognized = str(value).strip()
        text = recognized.lower()
        confidence = float(confidence)
        average = confidence if average is None else float(average)
        if not text or not math.isfinite(confidence) or not math.isfinite(average):
            return
        if text.startswith("type ") and len(text) > 5:
            # Dictation is free text the user can see and correct, so it is
            # scored on the average word instead of the weakest one; a single
            # shaky word should not throw away a whole sentence. Commands stay
            # strict because a misheard command acts on the desktop.
            if average < DICTATION_MIN_CONFIDENCE:
                return
            self.event_ready.emit(InteractionEvent(
                "voice", "dictation", "type_text", round(average, 3),
                {"text": recognized[5:].strip()}))
            return
        if text in ("left click", "click"):
            if average < LEFT_CLICK_MIN_CONFIDENCE:
                self.status.emit("Left-click phrase was unclear; please repeat ‘left click’")
                return
            self.event_ready.emit(InteractionEvent(
                "voice", "command", "click", round(average, 3),
                {"button": "left"}))
            return
        if text in ("space", "new line", "next line", "line break"):
            if average < KEY_COMMAND_MIN_CONFIDENCE:
                self.status.emit(
                    f"Didn’t catch ‘{text}’ clearly; please repeat the command")
                return
            key = "space" if text == "space" else "enter"
            self.event_ready.emit(InteractionEvent(
                "voice", "command", "press_key", round(average, 3),
                {"key": key}))
            return
        app_commands = {
            "open notepad": "notepad",
            "open note pad": "notepad",
            "start notepad": "notepad",
            "launch notepad": "notepad",
            "open calculator": "calculator",
            "start calculator": "calculator",
            "launch calculator": "calculator",
        }
        if text in app_commands:
            if average < OPEN_APP_MIN_CONFIDENCE:
                self.status.emit(
                    f"Didn’t catch ‘{text}’ clearly; please repeat the command")
                return
            name = app_commands[text]
            self.status.emit(f"Opening {name.title()}…")
            self.event_ready.emit(InteractionEvent(
                "voice", "command", "open_application", round(average, 3),
                {"name": name}))
            return
        if confidence < COMMAND_MIN_CONFIDENCE:
            if text in ("right click", "scroll up", "scroll down",
                        "space", "new line", "next line", "line break",
                        "pause accessflow", "stop accessflow",
                        "emergency pause"):
                self.status.emit(
                    f"Didn’t catch ‘{text}’ clearly; please repeat the command")
            return
        if text in ("pause accessflow", "stop accessflow", "emergency pause"):
            event = InteractionEvent("voice", "emergency_pause", "pause", confidence)
        elif text == "right click":
            event = InteractionEvent("voice", "command", "click", confidence, {"button": "right"})
        elif text == "scroll up":
            event = InteractionEvent("voice", "command", "scroll", confidence, {"amount": 3})
        elif text == "scroll down":
            event = InteractionEvent("voice", "command", "scroll", confidence, {"amount": -3})
        else:
            return
        self.event_ready.emit(event)

    def run(self):
        failure = None
        try:
            model = Model(self.model_directory)
            if self.stop_requested:
                # The session was paused while the model was still loading;
                # do not open the microphone for a stopped session.
                return
            recognizer = KaldiRecognizer(model, 16000)
            recognizer.SetWords(True)

            def callback(indata, frames, timing, status):
                if self.stop_requested:
                    return
                try:
                    self.audio.put_nowait(bytes(indata))
                except queue.Full:
                    self.dropped_blocks += 1
                    if not self.reported_backlog:
                        self.reported_backlog = True
                        self.status.emit(
                            "Voice recognition is behind; dropping audio "
                            "instead of delaying it")

            self.status.emit("Local voice input active")
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
                while not self.stop_requested:
                    try:
                        data = self.audio.get(timeout=.25)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        phrase = result.get("text", "")
                        words = result.get("result", [])
                        scores = [float(word.get("conf", 0.0)) for word in words if "conf" in word]
                        # Score on the words Vosk returned confidence for; a
                        # missing score is not a reason to discard a phrase.
                        if phrase and scores:
                            self._phrase(phrase, min(scores), sum(scores) / len(scores))
        except Exception as exc:
            failure = exc
        finally:
            # Report a load failure rather than replacing it with the generic
            # stopped message one line later.
            if failure is not None and not self.stop_requested:
                self.status.emit(f"Voice input stopped: {failure}")
            else:
                self.status.emit("Voice input stopped")

