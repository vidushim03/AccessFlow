# AccessFlow

Windows desktop MVP for adaptive multimodal assistive interaction. Python 3.11 is recommended.

## Start

    .venv\Scripts\python.exe main.py

Dependencies are listed in requirements.txt.

## Connected inputs

- Eye gaze, head tracking, and hand gestures share a single OpenCV camera capture worker. Face/hand models are fetched from official MediaPipe model storage on first camera start.
- Head tracking moves the pointer from the raw nose-tip position relative to a learned neutral pose, smoothed with a plain 0.18 EMA. The neutral pose is learned from a short stable window while the pointer is held still, so a bad first frame or sitting down mid-detection cannot bias the cursor. The pointer is anchored to wherever it already is, and re-anchors when head control takes over from another input, so starting a session, switching methods, or recentering never jumps the cursor. It is used when Eye gaze is off.
- Local voice commands and dictation use Vosk. Use the Download small English model button in Input methods, then enable voice. The model is stored under your local app data.
- Supported voice commands include left click, right click, scroll up/down, open Notepad/Calculator, and “type …”. “Pause AccessFlow” stops all active providers.
- All events pass through the confidence/source gate before reaching WindowsActions.
- pywinauto is used for named control clicks and window activation; PyAutoGUI provides coordinate mouse/keyboard actions.
- Escape is observed system-wide, so it pauses a running session even when another app has focus. The key is not consumed, so the focused app still receives it normally. Input aimed at elevated (administrator) windows cannot be observed. If Windows refuses to install the global hook (group policy, exhausted hook quota), AccessFlow says so in the session status, on the dashboard, and in a tray notification instead of quietly running without it; Escape then still pauses while an AccessFlow window has focus.
- The touchpad and mouse are never disabled. When real touchpad or mouse movement is detected, camera pointer output (including pinch clicks) steps aside for 1.5 seconds, then resumes automatically.
- The Tracking preview page shows the live mirrored camera, its landmarks, and a miniature screen with exactly where the current mapping would put the pointer. It never moves the real pointer, so hand, eye, and head mapping can be checked safely before starting a session. Whether or not it is calibrated, eye preview also shows the raw tracked position so the user can confirm the camera can see their eyes.
- Settings persist locally with QSettings. AccessFlow does not record raw audio or video.

Eye gaze requires the nine-point calibration wizard; pointer movement stays disabled until calibration is saved. The wizard fits a quadratic eye-to-screen model (constant, two iris offsets, their interaction, and their squares), which follows the eye's curved response near the screen edges better than a straight-line fit. Each point is also predicted from a mapping fitted on the other eight (leave-one-out), so a random or overfitted calibration is rejected instead of producing a pointer that wanders. Saved linear calibrations still load, with a notice recommending recalibration. Head tracking does not use the gaze calibration. Eye gaze and head tracking are mutually exclusive, so selecting one turns the other off. Calibration quality depends on camera position, lighting, and the individual user. Providers are opt-in. Use Escape from any application or Emergency pause to stop them; the UI pauses immediately and camera/audio workers finish shutting down in the background.


## Input stability

- Hand control is a relative trackpad mapping, not absolute fingertip aiming: only movement while the hand is visible moves the pointer. A light fingertip EMA (0.5) halves landmark jitter, a dead zone with a slow leak absorbs tremor without swallowing deliberate slow motion, and a per-frame 300 px cap rejects landmark glitches. The pointer freezes when the hand leaves the frame and only re-anchors on the next detection, so a hand returning elsewhere never teleports the cursor. Eye gaze keeps its One-Euro filter, small-motion dead zone, and jump limit. Head tracking instead uses the plain EMA on absolute position that keeps a still head from dragging the pointer. Brief hand-tracking dropouts do not immediately switch control to face tracking.
- Gaze calibration waits for the camera model, discards target-transition frames, and accepts a point only once the eye feature has settled (standard deviation below the reference tolerance). A failed attempt keeps the previous calibration, and a saved mapping records the screen size, gaze feature, and model, so a mapping made for another display or an older feature is refused instead of silently mis-aiming.
- Pinch uses a hand-size-relative distance, an 80 ms hold, separate enter/exit thresholds, and a click cooldown.
- Voice commands use Vosk word confidence and discard low-confidence recognition. Dictation is scored on the average word instead, because one shaky word should not discard a whole sentence.
- Typing and UI automation run on a dedicated action thread, so a long dictated phrase cannot freeze the window or delay Emergency pause. Pointer, click, and scroll stay synchronous because they take milliseconds.
- Pausing cancels that work: the action backlog is dropped and a phrase that is already being typed stops within about 64 ms (32 characters) instead of typing to the end. Quitting waits for the action thread, so the process never exits halfway through a keystroke and leaves a modifier held.
- A saved gaze mapping is validated (finite, and either the legacy 3×2 affine shape or the current 6×2 quadratic shape) before use, so a corrupted settings entry disables gaze rather than stopping the camera thread and taking head and hand tracking with it.
- Voice audio is buffered to a bounded queue; if recognition falls behind realtime, audio is dropped and reported instead of growing without limit.

Syntax, imports, UI construction, and the live head/hand/gaze camera paths have been validated on this device, including the Tracking preview render path. A live nine-point calibration and live microphone/voice behaviour still need validation on a person.

## Tests

    .venv\Scripts\python.exe -m unittest discover -s tests -v

`tests/test_safety.py` covers the pause path (backlog dropped, typing stopped mid-phrase, thread joined on quit), the confidence gate per action, disabled sources, failed-action reporting, and a failed hook install being reported. `tests/test_calibration.py` builds the real window offscreen and covers sample stability gating, the fit itself, preserving a working calibration when a recalibration fails, screen/feature/corruption validation, and malformed-matrix rejection. `tests/test_mapping.py` covers the still-hand/no-drift and reacquisition/no-jump rules, the hand dead zone, realistic fingertip jitter, quadratic gaze recovery and held-out rejection, and stable head-neutral learning. None needs a camera, a microphone, or a person.

- Camera output coordinates use PyAutoGUI’s physical-pixel screen size, matching Windows display scaling.
