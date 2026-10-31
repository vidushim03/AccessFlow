# AccessFlow

Windows desktop MVP for adaptive multimodal assistive interaction. Python 3.11 is recommended.

## Start

    .venv\Scripts\python.exe main.py

Dependencies are listed in requirements.txt.

## Connected inputs

- Eye gaze and hand gestures share a single OpenCV camera capture worker. Face/hand models are fetched from official MediaPipe model storage on first camera start.
- Local voice commands and dictation use Vosk. Use the Download small English model button in Input methods, then enable voice. The model is stored under your local app data.
- Supported voice commands include left click, right click, scroll up/down, open Notepad/Calculator, and “type …”. “Pause AccessFlow” stops all active providers.
- All events pass through the confidence/source gate before reaching WindowsActions.
- pywinauto is used for named control clicks and window activation; PyAutoGUI provides coordinate mouse/keyboard actions.
- Settings persist locally with QSettings. AccessFlow does not record raw audio or video.

Eye gaze requires the nine-point calibration wizard; pointer movement stays disabled until calibration is saved. Calibration quality depends on camera position, lighting, and the individual user. Providers are opt-in. Use Escape or Emergency pause to stop them.


## Input stability

- Camera pointer coordinates use speed-adaptive filtering, a small-motion dead zone, and per-frame movement limits. Brief hand-tracking dropouts do not immediately switch control to gaze.
- Gaze calibration waits for the camera model, ignores target-transition frames, takes a median reading per point, and rejects unstable or poor-fit results.
- Pinch uses a hand-size-relative distance, an 80 ms hold, separate enter/exit thresholds, and a click cooldown.
- Voice commands use Vosk word confidence; low-confidence recognition is discarded.

The changes have passed syntax and import checks. Live webcam, calibration, and microphone behavior still needs validation on this device.
- Camera output coordinates use PyAutoGUI’s physical-pixel screen size, matching Windows display scaling.
