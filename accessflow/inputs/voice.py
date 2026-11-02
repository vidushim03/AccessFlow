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
# Blocks waiting for the recogniser (8000 frames at 16 kHz = 0.5 s each).
# Recognition slower than realtime must drop audio, not grow without bound.
AUDIO_QUEUE_BLOCKS = 50


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
        text = value.lower().strip()
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
                "voice", "dictation", "type_text", round(average, 3), {"text": text[5:]}))
            return
        if confidence < COMMAND_MIN_CONFIDENCE:
            return
        if text in ("pause accessflow", "stop accessflow", "emergency pause"):
            event = InteractionEvent("voice", "emergency_pause", "pause", confidence)
        elif text in ("left click", "click"):
            event = InteractionEvent("voice", "command", "click", confidence, {"button": "left"})
        elif text == "right click":
            event = InteractionEvent("voice", "command", "click", confidence, {"button": "right"})
        elif text == "scroll up":
            event = InteractionEvent("voice", "command", "scroll", confidence, {"amount": 3})
        elif text == "scroll down":
            event = InteractionEvent("voice", "command", "scroll", confidence, {"amount": -3})
        elif text in ("open notepad", "open calculator"):
            event = InteractionEvent("voice", "command", "open_application", confidence, {"name": text[5:]})
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

