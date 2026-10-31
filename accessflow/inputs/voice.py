import json
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


class VoiceModelDownloadWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)
    status = Signal(str)

    def run(self):
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AccessFlow" / "models"
        base.mkdir(parents=True, exist_ok=True)
        archive = base / "vosk-model-small-en-us-0.15.zip"
        temporary = archive.with_suffix(".download")
        try:
            self.status.emit("Downloading offline speech model…")
            urllib.request.urlretrieve(MODEL_URL, temporary)
            temporary.replace(archive)
            with zipfile.ZipFile(archive) as zipped:
                for info in zipped.infolist():
                    target = (base / info.filename).resolve()
                    if os.path.commonpath((str(base.resolve()), str(target))) != str(base.resolve()):
                        raise ValueError("Model archive contains an unsafe path")
                zipped.extractall(base)
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
        self.audio = queue.Queue()

    def stop(self):
        self.stop_requested = True
        self.wait(3000)

    def _phrase(self, value):
        text = value.lower().strip()
        if text in ("pause accessflow", "stop accessflow", "emergency pause"):
            event = InteractionEvent("voice", "emergency_pause", "pause")
        elif text in ("left click", "click"):
            event = InteractionEvent("voice", "command", "click", .88, {"button": "left"})
        elif text == "right click":
            event = InteractionEvent("voice", "command", "click", .88, {"button": "right"})
        elif text == "scroll up":
            event = InteractionEvent("voice", "command", "scroll", .88, {"amount": 3})
        elif text == "scroll down":
            event = InteractionEvent("voice", "command", "scroll", .88, {"amount": -3})
        elif text in ("open notepad", "open calculator"):
            event = InteractionEvent("voice", "command", "open_application", .90, {"name": text[5:]})
        elif text.startswith("type "):
            event = InteractionEvent("voice", "dictation", "type_text", .82, {"text": text[5:]})
        else:
            return
        self.event_ready.emit(event)

    def run(self):
        try:
            model = Model(self.model_directory)
            recognizer = KaldiRecognizer(model, 16000)

            def callback(indata, frames, timing, status):
                if not self.stop_requested:
                    self.audio.put(bytes(indata))

            self.status.emit("Local voice input active")
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
                while not self.stop_requested:
                    try:
                        data = self.audio.get(timeout=.25)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(data):
                        phrase = json.loads(recognizer.Result()).get("text", "")
                        if phrase:
                            self._phrase(phrase)
        except Exception as exc:
            self.status.emit(f"Voice input stopped: {exc}")
        finally:
            self.status.emit("Voice input stopped")
